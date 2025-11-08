"""
Evaluation Metrics for Legal-DSL LLM
Calculate precision, recall, F1, ROUGE, and custom legal metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report


class ClauseEvaluator:
    """Evaluate clause extraction and classification"""
    
    @staticmethod
    def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Calculate precision, recall, F1 for clause classification
        
        Args:
            predictions: List of predicted clauses with labels
            ground_truth: List of ground truth clauses with labels
            
        Returns:
            Dict with precision, recall, F1 scores
        """
        if not predictions or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        pred_labels = [p['clause_type'] for p in predictions]
        true_labels = [g['clause_type'] for g in ground_truth]
        
        label_set = sorted(list(set(pred_labels + true_labels)))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', labels=label_set, zero_division=0
        )
        
        accuracy = sum([1 for p, t in zip(pred_labels, true_labels) if p == t]) / len(true_labels)
        
        cm = confusion_matrix(true_labels, pred_labels, labels=label_set)
        
        report = classification_report(true_labels, pred_labels, labels=label_set, output_dict=True, zero_division=0)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': report,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth)
        }
    
    @staticmethod
    def calculate_span_overlap_f1(predictions: List[Dict], ground_truth: List[Dict]) -> float:
        """
        Calculate F1 based on character span overlap
        Useful for clause boundary detection
        """
        pred_spans = set()
        for p in predictions:
            pred_spans.add((p['char_offset_start'], p['char_offset_end'], p['clause_type']))
        
        true_spans = set()
        for g in ground_truth:
            true_spans.add((g['char_offset_start'], g['char_offset_end'], g['clause_type']))
        
        if not pred_spans or not true_spans:
            return 0.0
        
        tp = len(pred_spans & true_spans)
        fp = len(pred_spans - true_spans)
        fn = len(true_spans - pred_spans)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1


class NER_Evaluator:
    """Evaluate named entity recognition"""
    
    @staticmethod
    def calculate_entity_f1(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """
        Calculate entity-level F1 scores
        
        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities
            
        Returns:
            Dict with overall and per-type F1 scores
        """
        pred_entities = set()
        for p in predictions:
            pred_entities.add((p['start_offset'], p['end_offset'], p['entity_type']))
        
        true_entities = set()
        for g in ground_truth:
            true_entities.add((g['start_offset'], g['end_offset'], g['entity_type']))
        
        if not pred_entities or not true_entities:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        tp = len(pred_entities & true_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        by_type = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for entity in pred_entities & true_entities:
            by_type[entity[2]]['tp'] += 1
        for entity in pred_entities - true_entities:
            by_type[entity[2]]['fp'] += 1
        for entity in true_entities - pred_entities:
            by_type[entity[2]]['fn'] += 1
        
        per_type_f1 = {}
        for entity_type, counts in by_type.items():
            p = counts['tp'] / (counts['tp'] + counts['fp']) if (counts['tp'] + counts['fp']) > 0 else 0
            r = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            per_type_f1[entity_type] = {'precision': p, 'recall': r, 'f1': f}
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'per_type': per_type_f1
        }


class SummarizationEvaluator:
    """Evaluate document summarization"""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict:
        """
        Calculate ROUGE scores
        
        Args:
            prediction: Generated summary
            reference: Ground truth summary
            
        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        scores = self.scorer.score(reference, prediction)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }
    
    @staticmethod
    def calculate_compression_ratio(original: str, summary: str) -> float:
        """Calculate compression ratio"""
        return len(summary) / len(original) if original else 0.0
    
    @staticmethod
    def evaluate_factual_consistency(summary: str, document: str, entities: List[Dict]) -> Dict:
        """
        Check if summary contains only facts from document
        Uses entity matching as proxy for factual consistency
        """
        summary_lower = summary.lower()
        
        entity_matches = 0
        entity_total = 0
        
        for entity in entities:
            entity_total += 1
            if entity['text'].lower() in summary_lower:
                entity_matches += 1
        
        consistency_score = entity_matches / entity_total if entity_total > 0 else 0.0
        
        return {
            'factual_consistency_score': consistency_score,
            'entities_preserved': entity_matches,
            'total_entities': entity_total
        }


class RAG_Evaluator:
    """Evaluate RAG system"""
    
    @staticmethod
    def calculate_grounding_accuracy(answer_data: Dict, ground_truth_chunks: List[int]) -> Dict:
        """
        Evaluate if retrieved chunks match ground truth relevant chunks
        
        Args:
            answer_data: RAG system output with source chunks
            ground_truth_chunks: List of chunk IDs that should be retrieved
            
        Returns:
            Dict with grounding metrics
        """
        retrieved_chunk_ids = [s['chunk_id'] for s in answer_data.get('sources', [])]
        
        if not retrieved_chunk_ids or not ground_truth_chunks:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        retrieved_set = set(retrieved_chunk_ids)
        truth_set = set(ground_truth_chunks)
        
        tp = len(retrieved_set & truth_set)
        fp = len(retrieved_set - truth_set)
        fn = len(truth_set - retrieved_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'retrieved_chunks': len(retrieved_chunk_ids),
            'relevant_chunks': len(ground_truth_chunks),
            'overlap': tp
        }
    
    @staticmethod
    def calculate_mrr(answer_data: Dict, relevant_chunk_id: int) -> float:
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            answer_data: RAG output with ranked sources
            relevant_chunk_id: ID of the most relevant chunk
            
        Returns:
            MRR score
        """
        retrieved_chunk_ids = [s['chunk_id'] for s in answer_data.get('sources', [])]
        
        if relevant_chunk_id in retrieved_chunk_ids:
            rank = retrieved_chunk_ids.index(relevant_chunk_id) + 1
            return 1.0 / rank
        else:
            return 0.0


def run_full_evaluation(predictions: Dict, ground_truth: Dict) -> Dict:
    """
    Run complete evaluation across all tasks
    
    Args:
        predictions: Dict with predictions for all tasks
        ground_truth: Dict with ground truth for all tasks
        
    Returns:
        Comprehensive evaluation results
    """
    results = {}
    
    if 'clauses' in predictions and 'clauses' in ground_truth:
        clause_eval = ClauseEvaluator()
        results['clause_classification'] = clause_eval.calculate_metrics(
            predictions['clauses'], ground_truth['clauses']
        )
    
    if 'entities' in predictions and 'entities' in ground_truth:
        ner_eval = NER_Evaluator()
        results['ner'] = ner_eval.calculate_entity_f1(
            predictions['entities'], ground_truth['entities']
        )
    
    if 'summary' in predictions and 'summary' in ground_truth:
        sum_eval = SummarizationEvaluator()
        results['summarization'] = sum_eval.calculate_rouge(
            predictions['summary'], ground_truth['summary']
        )
    
    return results
