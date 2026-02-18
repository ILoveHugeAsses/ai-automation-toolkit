"""
Sentiment Analysis Tool - Production Ready
Author: v0id-lab
License: MIT

Analyze sentiment of text using multiple methods:
- VADER (rule-based, no API needed)
- TextBlob (simple ML-based)
- HuggingFace transformers (advanced)
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class SentimentAnalyzer:
    def __init__(self, method: str = "vader"):
        """
        Initialize sentiment analyzer
        
        Args:
            method: "vader", "textblob", or "huggingface"
        """
        self.method = method.lower()
        
        if self.method == "vader":
            if not VADER_AVAILABLE:
                raise ImportError("Install: pip install vaderSentiment")
            self.analyzer = SentimentIntensityAnalyzer()
        
        elif self.method == "textblob":
            if not TEXTBLOB_AVAILABLE:
                raise ImportError("Install: pip install textblob")
        
        elif self.method == "huggingface":
            if not REQUESTS_AVAILABLE:
                raise ImportError("Install: pip install requests")
            self.api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not self.api_key:
                raise ValueError("HUGGINGFACE_API_KEY not set")
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Dict with sentiment scores and label
        """
        if self.method == "vader":
            return self._analyze_vader(text)
        elif self.method == "textblob":
            return self._analyze_textblob(text)
        elif self.method == "huggingface":
            return self._analyze_huggingface(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _analyze_vader(self, text: str) -> Dict:
        """VADER sentiment analysis"""
        scores = self.analyzer.polarity_scores(text)
        
        # Determine label
        compound = scores['compound']
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "method": "vader",
            "label": label,
            "scores": {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu'],
                "compound": compound
            },
            "confidence": abs(compound),
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_textblob(self, text: str) -> Dict:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine label
        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            "method": "textblob",
            "label": label,
            "scores": {
                "polarity": polarity,
                "subjectivity": subjectivity
            },
            "confidence": abs(polarity),
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_huggingface(self, text: str) -> Dict:
        """HuggingFace sentiment analysis"""
        url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {"inputs": text}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                scores = result[0]
                
                # Find highest score
                top = max(scores, key=lambda x: x['score'])
                
                return {
                    "method": "huggingface",
                    "label": top['label'],
                    "scores": {item['label']: item['score'] for item in scores},
                    "confidence": top['score'],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise ValueError("Unexpected response format")
                
        except Exception as e:
            raise Exception(f"HuggingFace analysis failed: {e}")
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts"""
        return [self.analyze(text) for text in texts]
    
    def analyze_file(self, filepath: str, output: Optional[str] = None) -> List[Dict]:
        """
        Analyze text file (one text per line)
        
        Args:
            filepath: Input file path
            output: Optional output JSON file
            
        Returns:
            List of analysis results
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = self.analyze_batch(texts)
        
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def get_summary(self, results: List[Dict]) -> Dict:
        """Get summary statistics from batch results"""
        if not results:
            return {}
        
        labels = [r['label'] for r in results]
        
        summary = {
            "total": len(results),
            "positive": labels.count("POSITIVE"),
            "negative": labels.count("NEGATIVE"),
            "neutral": labels.count("NEUTRAL"),
            "avg_confidence": sum(r['confidence'] for r in results) / len(results)
        }
        
        summary['positive_pct'] = (summary['positive'] / summary['total']) * 100
        summary['negative_pct'] = (summary['negative'] / summary['total']) * 100
        summary['neutral_pct'] = (summary['neutral'] / summary['total']) * 100
        
        return summary


# ========== CLI USAGE ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sentiment Analysis Tool")
    parser.add_argument("--method", default="vader", choices=["vader", "textblob", "huggingface"])
    parser.add_argument("--text", help="Single text to analyze")
    parser.add_argument("--file", help="File with texts (one per line)")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    try:
        analyzer = SentimentAnalyzer(method=args.method)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)
    
    if args.text:
        # Single text analysis
        result = analyzer.analyze(args.text)
        print(json.dumps(result, indent=2))
        
    elif args.file:
        # File analysis
        results = analyzer.analyze_file(args.file, args.output)
        summary = analyzer.get_summary(results)
        
        print(f"\n📊 Sentiment Analysis Summary ({args.method})")
        print(f"Total texts: {summary['total']}")
        print(f"Positive: {summary['positive']} ({summary['positive_pct']:.1f}%)")
        print(f"Negative: {summary['negative']} ({summary['negative_pct']:.1f}%)")
        print(f"Neutral: {summary['neutral']} ({summary['neutral_pct']:.1f}%)")
        print(f"Avg confidence: {summary['avg_confidence']:.2f}")
        
        if args.output:
            print(f"\nResults saved to: {args.output}")
    
    else:
        print("Error: Provide --text or --file")
        parser.print_help()
