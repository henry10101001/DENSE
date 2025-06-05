from trl import BasePairwiseJudge
from openai import OpenAI
import math
from typing import List
import os


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14" # most recent snapshot as of 06/04/2025


class CustomJudge(BasePairwiseJudge):
    """
    A pairwise judge that combines length-based scoring and OpenAI-based quality assessment
    to determine which of two completions is better.
    """
    
    def __init__(self, model: str = JUDGE_MODEL, length_weight: float = 0.3, quality_weight: float = 0.7):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.length_weight = length_weight
        self.quality_weight = quality_weight
        
        self.judging_prompt = """
        You are a helpful assistant and a grammar and coherence expert that is tasked with assessing whether a response maintains the same meaning as the prompt.
        The Prompt is typically a normal chunk of text, and the Response is a re-worded version of the prompt.
        The Response should retain the same meaning as the Prompt, and should be grammatically correct and coherent.
        The Response should not add to, or remove from, the information in the Prompt.
        
        Your task is to judge the quality of the response in terms of retained meaning (most important), coherence, and grammar/spelling.
        The scores should be between 0.00 and 1.00, with 1.00 being the highest quality. The scores should only evaluate the Response, not the Prompt.
        
        Please respond in this format:
        ```
        MEANING_SCORE: <0.00-1.00>
        COHERENCE_SCORE: <0.00-1.00>
        GRAMMAR_SCORE: <0.00-1.00>
        ```
        
        For example:
        ```
        MEANING_SCORE: 0.78
        COHERENCE_SCORE: 0.92
        GRAMMAR_SCORE: 0.89
        ```
        (Do not include any other text or formatting in your response. Do not even include the backticks (```) in your response.)
        
        
        Now, please evaluate the following Response in terms of retained meaning, coherence, and grammar/spelling, in relation to its corresponding Prompt.
        
        Prompt: {prompt}
        Response: {completion}
        
        Please evaluate now, and respond with just the scores, no other text.
        """
    
    def judge(self, prompts, completions, shuffle_order=False):
        """
        Judge pairs of completions and return 0 or 1 indicating which completion is better.
        Returns 0 if first completion is better, 1 if second completion is better.
        """
        results = []
        
        for prompt, completion_pair in zip(prompts, completions):
            completion_0, completion_1 = completion_pair
            
            # get scores for both completions
            score_0 = self._score_completion(prompt, completion_0)
            score_1 = self._score_completion(prompt, completion_1)
            
            # return 0 if first completion is better, 1 if second is better
            results.append(0 if score_0 > score_1 else 1)
        
        return results
    
    def _score_completion(self, prompt: str, completion: str) -> float:
        """Score a single completion using both length and quality metrics"""
        # length scoring component
        length_score = self._get_length_score(prompt, completion)
        
        # quality scoring component (using OpenAI API)
        quality_score = self._get_quality_score(prompt, completion)
        
        # combine scores with weights
        total_score = (self.length_weight * length_score) + (self.quality_weight * quality_score)
        
        return total_score # should be between 0 and 1
    
    def _get_length_score(self, prompt: str, completion: str) -> float:
        """Score based on length optimization (shorter is better)"""
        prompt_length = len(prompt.split())
        completion_length = len(completion.split())
        difference = prompt_length - completion_length  # positive for shorter completions
        
        # score as a percentage of the length improvement
        if prompt_length > 0:
            score = difference / prompt_length
        else:
            score = 0.0
        
        # normalize to [0, 1] range using sigmoid
        score = 1 / (1 + math.exp(-score))
        
        return score
    
    def _get_quality_score(self, prompt: str, completion: str) -> float:
        """Get quality score from OpenAI assessment"""
        try:
            judge_prompt = self.judging_prompt.format(prompt=prompt, completion=completion)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_completion_tokens=50,  # example is only ~30 tokens
            )
            
            score_text = response.choices[0].message.content.strip()
            
            try:
                meaning_score = float(score_text.split("MEANING_SCORE:")[1].split("\n")[0].strip())
                coherence_score = float(score_text.split("COHERENCE_SCORE:")[1].split("\n")[0].strip())
                grammar_score = float(score_text.split("GRAMMAR_SCORE:")[1].split("\n")[0].strip())
                
                # clamp to 0-1 range
                meaning_score = max(0.0, min(1.0, meaning_score))
                coherence_score = max(0.0, min(1.0, coherence_score))
                grammar_score = max(0.0, min(1.0, grammar_score))
                
                # return the average of the scores
                return (meaning_score + coherence_score + grammar_score) / 3.0
            except (ValueError, IndexError):
                return 0.5  # neutral score on parsing error
        except Exception as e:
            print(f"Error getting quality score: {e}")
            return 0.5  # neutral score on error