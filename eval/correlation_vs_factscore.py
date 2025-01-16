"""Calculates correlation between SAFE and human ratings from FActScore using Perplexity API.

Run command:
```
python -m eval.correlation_vs_factscore \
    --samples=100 \
    --save_results=True
```
"""

import datetime
import json
import os
import logging
from typing import Any, Dict, List, Tuple
import re  # Add at the top with other imports

from openai import OpenAI
from absl import app
from absl import flags
import langfun as lf
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

from common import utils
from common import shared_config
from eval import metric_utils

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

_SAMPLES = flags.DEFINE_integer(
    'samples', default=-1, help='Number of samples to eval.'
)
_SAVE_RESULTS = flags.DEFINE_boolean(
    'save_results', default=True, help='Whether to save all results to a JSON.'
)

_FACTSCORE_DATA_FOLDER = os.path.join(
    shared_config.root_dir, 'third_party/factscore/labeled_data/'
)
_DATE_AND_TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# Constants for labels
SUPPORTED_LABEL = 'S'
IRRELEVANT_LABEL = 'IR' 
NOT_SUPPORTED_LABEL = 'NS'

class PerplexityFactChecker:
    """Uses Perplexity API for fact verification."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    def verify_fact(self, fact: str, context: str, human_label: str = 'unknown') -> str:
        """Verify if a fact is supported by the context.
        
        Args:
            fact: The fact to verify
            context: The context to verify against
            human_label: The human-assigned label (for logging)
            
        Returns:
            One of: SUPPORTED_LABEL, NOT_SUPPORTED_LABEL, or ERROR
        """
        logging.info(f"\nVerifying fact: {fact}")
        logging.info(f"Human label: {human_label}")
        logging.info(f"Context: {context}")
        
        verification_messages = [
            {
                "role": "system",
                "content": """You are a precise fact verifier. Your job is to determine if a fact is SUPPORTED by your research, using the context only to clarify what the fact is referring to.

A fact is SUPPORTED (S) if:
1. You can verify the fact is true through your research
2. The context helps clarify any ambiguous references (like pronouns)

A fact is NOT SUPPORTED (NS) if:
1. You find contradicting information in your research
2. The fact is too ambiguous even with context
3. You cannot verify the fact through research

Note: The context helps understand what the fact is referring to, but don't trust the context for verification - use your research.

You must respond with EXACTLY one of these two letters:
- S
- NS

DO NOT include any other text, explanations, or citations."""
            },
            {
                "role": "user",
                "content": f"Determine if this fact is supported by the context:\n\nFact: {fact}\nContext: {context}"
            }
        ]
        
        try:
            verification_response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=verification_messages,
            )
            result = verification_response.choices[0].message.content.strip().upper()
            
            # Extract just the label using regex - match either NS or S
            match = re.match(r'^(NS|S)', result)
            if match:
                result = match.group(1)
                if result in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:
                    logging.info(f"Machine label: {result}")
                    logging.info(f"Label comparison - Human: {human_label}, Machine: {result}")
                    return result
            
            logging.warning(f"Invalid support response: {result}, marking as ERROR")
            result = 'ERROR'
            
            logging.info(f"Machine label: {result}")
            logging.info(f"Label comparison - Human: {human_label}, Machine: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Error during fact verification: {e}")
            logging.info(f"Label comparison - Human: {human_label}, Machine: ERROR (error case)")
            return 'ERROR'

    def process_facts(self, facts: List[Dict[str, str]], context: str) -> Dict[str, Any]:
        """Process a list of facts to verify them."""
        logging.info("Starting fact verification")
        results = {
            SUPPORTED_LABEL: 0,
            NOT_SUPPORTED_LABEL: 0,
            'ERROR': 0,
            'atomic_facts': []
        }
        
        # Filter out irrelevant facts
        relevant_facts = [f for f in facts if f.get('label') != IRRELEVANT_LABEL]
        
        for i, fact in enumerate(relevant_facts, 1):
            logging.info(f"Processing fact {i}/{len(relevant_facts)}")
            human_label = fact.get('label', 'unknown')
            label = self.verify_fact(fact['text'], context, human_label)
            results[label] += 1
            results['atomic_facts'].append({
                'text': fact['text'],
                'label': label,
                'human_label': human_label,
                'source_text': fact.get('source_text', '')
            })
        
        logging.info(f"Processing complete. Results: {results}")
        return results

def load_factscore_data(filepath: str) -> List[Dict]:
    """Load data from FActScore JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Parse the JSON string into a dictionary
            item = json.loads(line)
            data.append(item)
    return data

def calculate_metrics(y_true: List[str], y_pred: List[str], label: str) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score for a specific label."""
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    false_positives = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
    false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': sum(1 for t in y_true if t == label)
    }

def compute_detailed_metrics(human_facts: List[Dict[str, Any]], 
                           perplexity_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes detailed comparison metrics between human and model ratings."""
    # Filter out irrelevant facts
    relevant_pairs = [(h, p) for h, p in zip(human_facts, perplexity_results['atomic_facts']) 
                     if h['label'] != IRRELEVANT_LABEL]
    
    if not relevant_pairs:
        return {
            'overall': {'total_facts': 0, 'accuracy': 0.0},
            'per_label': {
                SUPPORTED_LABEL: {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0},
                NOT_SUPPORTED_LABEL: {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0}
            },
            'disagreements': []
        }
    
    human_facts_filtered, perplexity_facts_filtered = zip(*relevant_pairs)
    
    y_true = [fact['label'] for fact in human_facts_filtered]
    y_pred = [fact['label'] for fact in perplexity_facts_filtered]
    
    # Calculate accuracy only for valid comparisons (where both labels are known)
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != 'unknown' and p != 'unknown']
    if valid_pairs:
        accuracy = sum(1 for t, p in valid_pairs if t == p) / len(valid_pairs)
    else:
        accuracy = 0.0
    
    metrics = {
        'overall': {
            'total_facts': len(valid_pairs),
            'accuracy': accuracy
        },
        'per_label': {
            SUPPORTED_LABEL: calculate_metrics(y_true, y_pred, SUPPORTED_LABEL),
            NOT_SUPPORTED_LABEL: calculate_metrics(y_true, y_pred, NOT_SUPPORTED_LABEL)
        },
        'confusion_matrix': {
            'true_labels': y_true,
            'predicted_labels': y_pred
        }
    }
    
    # Add disagreement examples with source context (excluding irrelevant facts)
    metrics['disagreements'] = [
        {
            'text': h_fact['text'],
            'human_label': h_fact['label'],
            'model_label': p_fact['label'],
            'source_text': h_fact.get('source_text', '')
        }
        for h_fact, p_fact in zip(human_facts_filtered, perplexity_facts_filtered)
        if h_fact['label'] != p_fact['label'] and h_fact['label'] != 'unknown'
    ]
    
    return metrics

def compute_correlation(human_scores: List[Dict[str, int]], 
                       perplexity_scores: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Compute correlation between human and Perplexity API scores."""
    results = {}
    for metric in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL]:  # Removed IRRELEVANT_LABEL
        human_values = [score[metric] for score in human_scores]
        perplexity_values = [score[metric] for score in perplexity_scores]
        
        logging.info(f"\nMetric: {metric}")
        logging.info(f"Human values: {human_values[:5]}...")
        logging.info(f"Perplexity values: {perplexity_values[:5]}...")
        
        try:
            pearson = stats.pearsonr(human_values, perplexity_values)
            spearman = stats.spearmanr(human_values, perplexity_values)
            
            results[metric] = {
                'pearson': {
                    'correlation': pearson.statistic,
                    'p_value': pearson.pvalue
                },
                'spearman': {
                    'correlation': spearman.statistic,
                    'p_value': spearman.pvalue
                }
            }
        except Exception as e:
            logging.error(f"Error computing correlation for {metric}: {e}")
            results[metric] = {
                'pearson': {'correlation': float('nan'), 'p_value': float('nan')},
                'spearman': {'correlation': float('nan'), 'p_value': float('nan')}
            }
    
    return results

def main(_):
    logging.info("Starting evaluation")
    
    # Initialize Perplexity client
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        logging.error("PERPLEXITY_API_KEY environment variable not set")
        return
    fact_checker = PerplexityFactChecker(api_key)
    
    # Load FActScore data
    all_data = []
    for filename in os.listdir(_FACTSCORE_DATA_FOLDER):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(_FACTSCORE_DATA_FOLDER, filename)
            all_data.extend(load_factscore_data(filepath))
    
    if _SAMPLES.value > 0:
        all_data = all_data[:_SAMPLES.value]
    logging.info(f"Processing {len(all_data)} samples")
    
    # Process data using Perplexity API
    results = []
    
    for i, item in enumerate(all_data, 1):
        logging.info(f"\nProcessing item {i}/{len(all_data)}")
        
        # Use the full output text as context for all facts
        context = item['output']
        
        # Process each annotation
        for annotation in item['annotations']:
            # Get human atomic facts for this annotation, excluding irrelevant facts
            human_facts = [fact for fact in annotation['human-atomic-facts'] 
                          if fact['label'] != IRRELEVANT_LABEL]
            
            if not human_facts:  # Skip if no relevant facts
                continue
                
            # Verify facts using Perplexity API with the full output text as context
            perplexity_result = fact_checker.process_facts(human_facts, context)
            
            # Track all cases for analysis (irrelevant facts already filtered)
            for h_fact, p_fact in zip(human_facts, perplexity_result['atomic_facts']):
                results.append({
                    'text': h_fact['text'],
                    'human_label': h_fact['label'],
                    'perplexity_label': p_fact['label'],
                    'source_text': annotation['text'],
                    'full_context': context,
                    'is_relevant': annotation['is-relevant']
                })
        
        # Save intermediate results every 10 items
        if i % 10 == 0:
            intermediate_results = {
                'date_and_time': _DATE_AND_TIME,
                'samples_processed': i,
                'results': results
            }
            intermediate_path = os.path.join(
                shared_config.path_to_result,
                f'intermediate_results_{_DATE_AND_TIME}_sample_{i}.json'
            )
            os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
            with open(intermediate_path, 'w') as f:
                json.dump(intermediate_results, f, indent=2)
            logging.info(f"Saved intermediate results to {intermediate_path}")
    
    # Save detailed disagreements to a separate file
    disagreements_path = os.path.join(
        shared_config.path_to_result,
        f'detailed_disagreements_{_DATE_AND_TIME}.json'
    )
    disagreements = [
        result for result in results 
        if result['human_label'] != result['perplexity_label']
    ]
    with open(disagreements_path, 'w') as f:
        json.dump({
            'date_and_time': _DATE_AND_TIME,
            'total_disagreements': len(disagreements),
            'disagreements': disagreements
        }, f, indent=2)
    logging.info(f"Saved detailed disagreements to {disagreements_path}")
    
    # Compute overall metrics
    all_human_facts = []
    all_perplexity_results = {'atomic_facts': []}
    
    for result in results:
        if result['is_relevant']:  # Only include facts from relevant annotations
            all_human_facts.append({
                'text': result['text'],
                'label': result['human_label'],
                'source_text': result['source_text']
            })
            all_perplexity_results['atomic_facts'].append({
                'text': result['text'],
                'label': result['perplexity_label']
            })
    
    detailed_metrics = compute_detailed_metrics(all_human_facts, all_perplexity_results)
    
    # Save final results
    if _SAVE_RESULTS.value:
        final_results = {
            'date_and_time': _DATE_AND_TIME,
            'samples': len(all_data),
            'detailed_metrics': detailed_metrics,
            'results': results
        }
        
        out_folder = shared_config.path_to_result
        out_path = os.path.join(
            out_folder,
            f'perplexity_evaluation_results_{_DATE_AND_TIME}.json'
        )
        
        os.makedirs(out_folder, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logging.info(f"Results saved to {out_path}")
    
    # Print summary metrics
    print('\nEvaluation Results:')
    print(f'Total samples processed: {len(all_data)}')
    print(f'Total facts evaluated: {detailed_metrics["overall"]["total_facts"]}')
    print(f'Overall accuracy: {detailed_metrics["overall"]["accuracy"]:.3f}')
    
    print('\nPer-Label Metrics:')
    for label, metrics in detailed_metrics['per_label'].items():
        if label != IRRELEVANT_LABEL:  # Skip irrelevant metrics
            print(f'\n{label}:')
            print(f'  Precision: {metrics["precision"]:.3f}')
            print(f'  Recall: {metrics["recall"]:.3f}')
            print(f'  F1: {metrics["f1"]:.3f}')
            print(f'  Support: {metrics["support"]}')
    
    print(f'\nNumber of Disagreement Cases: {len(detailed_metrics["disagreements"])}')
    
    # Print some example disagreements
    print('\nExample Disagreements:')
    for i, case in enumerate(detailed_metrics['disagreements'][:5], 1):
        print(f'\n{i}. Fact: {case["text"]}')
        print(f'   Human Label: {case["human_label"]}')
        print(f'   Model Label: {case["model_label"]}')
        print(f'   Source Text: {case["source_text"]}')

if __name__ == '__main__':
    app.run(main)
