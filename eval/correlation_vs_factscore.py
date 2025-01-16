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

from openai import OpenAI
from absl import app
from absl import flags
import langfun as lf
from matplotlib import pyplot as plt
from scipy import stats

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
    """Uses Perplexity API for atomic fact extraction and verification."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
    def extract_atomic_facts(self, text: str) -> List[str]:
        """Extract atomic facts from text using Perplexity API."""
        messages = [
            {
                "role": "system",
                "content": """You are a precise fact extractor. Extract atomic facts from the given text and return them as a JSON array of strings.
Each fact should be:
1. A single, simple statement
2. Self-contained and independent
3. Objective and verifiable
4. Not a subjective interpretation

Return ONLY a JSON array of strings. No additional text, no code block formatting."""
            },
            {
                "role": "user",
                "content": f"Extract atomic facts from this text. Return ONLY a JSON array of strings:\n\n{text}"
            }
        ]
        
        try:
            logging.info(f"Extracting facts from text: {text[:100]}...")
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages,
            )
            content = response.choices[0].message.content.strip()
            logging.info(f"API Response: {content}")
            
            # Remove any markdown code block formatting
            if content.startswith("```"):
                content = "\n".join(content.split("\n")[1:-1])  # Remove first and last lines
            content = content.strip()
            
            facts = json.loads(content)
            if isinstance(facts, list):
                logging.info(f"Successfully extracted {len(facts)} facts")
                return facts
            logging.warning("API response was not a list of facts")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response as JSON: {e}")
            logging.error(f"Content that failed to parse: {content}")
            return []
        except Exception as e:
            logging.error(f"Error during fact extraction: {e}")
            return []

    def verify_fact(self, fact: str, context: str) -> str:
        """Verify a single atomic fact using Perplexity API."""
        messages = [
            {
                "role": "system",
                "content": """You are a precise fact verifier. You must return EXACTLY ONE of these labels with no additional text:
S = The fact is fully supported by the context
NS = The fact contradicts or is not supported by the context
IR = The fact is irrelevant or cannot be verified using the context

Return ONLY the label (S, NS, or IR). No other text or explanation."""
            },
            {
                "role": "user",
                "content": f"Verify this fact against the context. Return ONLY one label (S, NS, or IR):\n\nFact: {fact}\nContext: {context}"
            }
        ]
        
        try:
            logging.info(f"Verifying fact: {fact}")
            response = self.client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=messages,
            )
            result = response.choices[0].message.content.strip()
            result = result.split("\n")[0].strip()  # Take only first line
            result = result.split(" ")[0].strip()   # Take only first word
            
            # Extract base label (S, NS, or IR) from response with brackets
            base_label = result.split("[")[0].strip()
            
            logging.info(f"Verification result: {result}")
            if base_label in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL, IRRELEVANT_LABEL]:
                return base_label
            logging.warning(f"Invalid label returned: {result}, defaulting to NS")
            return NOT_SUPPORTED_LABEL
        except Exception as e:
            logging.error(f"Error during fact verification: {e}")
            return NOT_SUPPORTED_LABEL

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text to extract and verify atomic facts."""
        logging.info("Starting text processing")
        facts = self.extract_atomic_facts(text)
        results = {
            SUPPORTED_LABEL: 0,
            NOT_SUPPORTED_LABEL: 0,
            IRRELEVANT_LABEL: 0,
            'atomic_facts': []
        }
        
        for i, fact in enumerate(facts, 1):
            logging.info(f"Processing fact {i}/{len(facts)}")
            label = self.verify_fact(fact, text)
            results[label] += 1
            results['atomic_facts'].append({
                'text': fact,
                'label': label
            })
        
        logging.info(f"Processing complete. Results: {results}")
        return results

def load_factscore_data(input_path: str) -> List[Dict[str, Any]]:
    """Loads FActScore data from a file."""
    logging.info(f"Loading data from {input_path}")
    result = []
    for data in utils.read_from_jsonlines(input_path):
        if 'input' not in data or 'output' not in data:
            continue
            
        result.append({
            'model_name': os.path.basename(input_path).split('.')[0],
            'prompt': data['input'],
            'response': data['output'],
            'annotations': data.get('annotations', [])  # Include annotations for human scores
        })
    logging.info(f"Loaded {len(result)} samples from {input_path}")
    return result

def compute_correlation(human_scores: List[Dict[str, int]], 
                       perplexity_scores: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """Compute correlation between human and Perplexity API scores."""
    results = {}
    for metric in [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL, IRRELEVANT_LABEL]:
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

def compute_detailed_metrics(human_scores: List[Dict[str, Any]], 
                           perplexity_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes detailed comparison metrics between human and model ratings."""
    metrics = {
        'overall': {
            'total_facts': 0,
            'total_agreements': 0,
            'agreement_rate': 0.0
        },
        'per_category': {
            SUPPORTED_LABEL: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            NOT_SUPPORTED_LABEL: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
            IRRELEVANT_LABEL: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
        },
        'disagreement_analysis': []
    }
    
    # Calculate per-category metrics
    for h, p in zip(human_scores, perplexity_scores):
        h_facts = {fact['text']: fact['label'] for fact in h['atomic_facts']}
        p_facts = {fact['text']: fact['label'] for fact in p['atomic_facts']}
        
        metrics['overall']['total_facts'] += len(h_facts)
        
        for text, h_label in h_facts.items():
            if text in p_facts:
                p_label = p_facts[text]
                if h_label == p_label:
                    metrics['overall']['total_agreements'] += 1
                    metrics['per_category'][h_label]['true_positives'] += 1
                else:
                    metrics['per_category'][h_label]['false_negatives'] += 1
                    metrics['per_category'][p_label]['false_positives'] += 1
                    metrics['disagreement_analysis'].append({
                        'text': text,
                        'human_label': h_label,
                        'model_label': p_label
                    })
    
    # Calculate agreement rate
    if metrics['overall']['total_facts'] > 0:
        metrics['overall']['agreement_rate'] = (
            metrics['overall']['total_agreements'] / metrics['overall']['total_facts']
        ) * 100
    
    # Calculate precision, recall, F1 for each category
    for category in metrics['per_category']:
        cat_metrics = metrics['per_category'][category]
        tp = cat_metrics['true_positives']
        fp = cat_metrics['false_positives']
        fn = cat_metrics['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        cat_metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return metrics

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
    human_scores = []
    perplexity_scores = []
    disagreement_cases = []
    
    for i, item in enumerate(all_data, 1):
        logging.info(f"\nProcessing item {i}/{len(all_data)}")
        
        # Get human scores from original annotations
        human_result = {
            SUPPORTED_LABEL: 0,
            NOT_SUPPORTED_LABEL: 0,
            IRRELEVANT_LABEL: 0,
            'atomic_facts': []
        }
        
        if 'annotations' in item:
            for annotation in item['annotations']:
                if 'human-atomic-facts' in annotation:
                    for fact in annotation['human-atomic-facts']:
                        if 'label' in fact:
                            human_result[fact['label']] += 1
                            human_result['atomic_facts'].append({
                                'text': fact.get('text', ''),
                                'label': fact['label']
                            })
        
        logging.info(f"Human scores for item {i}: {human_result}")
        human_scores.append(human_result)
        
        # Get Perplexity API scores
        perplexity_result = fact_checker.process_text(item['response'])
        logging.info(f"Perplexity scores for item {i}: {perplexity_result}")
        perplexity_scores.append(perplexity_result)
        
        # Track disagreement cases
        for h_fact in human_result['atomic_facts']:
            for p_fact in perplexity_result['atomic_facts']:
                if h_fact['text'] == p_fact['text'] and h_fact['label'] != p_fact['label']:
                    disagreement_cases.append({
                        'text': h_fact['text'],
                        'human_label': h_fact['label'],
                        'perplexity_label': p_fact['label'],
                        'context': item['response']
                    })
        
        # Save intermediate results every 10 items
        if i % 10 == 0:
            intermediate_results = {
                'date_and_time': _DATE_AND_TIME,
                'samples_processed': i,
                'human_scores': human_scores,
                'perplexity_scores': perplexity_scores,
                'disagreement_cases': disagreement_cases
            }
            intermediate_path = os.path.join(
                shared_config.path_to_result,
                f'intermediate_results_{_DATE_AND_TIME}_sample_{i}.json'
            )
            os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
            with open(intermediate_path, 'w') as f:
                json.dump(intermediate_results, f, indent=2)
            logging.info(f"Saved intermediate results to {intermediate_path}")
    
    # Compute correlation and detailed metrics
    correlation_results = compute_correlation(human_scores, perplexity_scores)
    detailed_metrics = compute_detailed_metrics(human_scores, perplexity_scores)
    
    # Save results
    if _SAVE_RESULTS.value:
        results = {
            'date_and_time': _DATE_AND_TIME,
            'samples': len(all_data),
            'correlation_results': correlation_results,
            'detailed_metrics': detailed_metrics,
            'human_scores': human_scores,
            'perplexity_scores': perplexity_scores,
            'disagreement_cases': disagreement_cases
        }
        
        out_folder = shared_config.path_to_result
        out_path = os.path.join(
            out_folder,
            f'perplexity_correlation_results_{_DATE_AND_TIME}.json'
        )
        
        os.makedirs(out_folder, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {out_path}")
    
    # Print results
    print('\nCorrelation Results:')
    for metric in correlation_results:
        print(f'\n{metric}:')
        for corr_type, values in correlation_results[metric].items():
            print(f'  {corr_type.capitalize()}:')
            print(f'    Correlation: {values["correlation"]:.3f}')
            print(f'    P-value: {values["p_value"]:.3f}')
    
    print('\nDetailed Metrics:')
    print(f'Overall Agreement Rate: {detailed_metrics["overall"]["agreement_rate"]:.1f}%')
    print(f'Total Facts: {detailed_metrics["overall"]["total_facts"]}')
    print(f'Total Agreements: {detailed_metrics["overall"]["total_agreements"]}')
    
    print('\nPer-Category Metrics:')
    for category in detailed_metrics['per_category']:
        metrics = detailed_metrics['per_category'][category]
        print(f'\n{category}:')
        print(f'  Precision: {metrics["precision"]:.3f}')
        print(f'  Recall: {metrics["recall"]:.3f}')
        print(f'  F1: {metrics["f1"]:.3f}')
    
    print(f'\nNumber of Disagreement Cases: {len(detailed_metrics["disagreement_analysis"])}')

if __name__ == '__main__':
    app.run(main)
