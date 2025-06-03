import re
import requests
import openai
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import math


class BaseJudge(ABC):
    """Base class for all judges"""
    
    @abstractmethod
    def judge(self, prompts: List[str], completions: List[List[str]], **kwargs) -> List[float]:
        """
        Judge the quality of completions for given prompts.
        
        Args:
            prompts: List of input prompts
            completions: List of completion lists (each prompt can have multiple completions)
            
        Returns:
            List of scores (higher = better quality)
        """
        pass


class LengthJudge(BaseJudge):
    """
    A simple judge that checks if the length of the completion is within a reasonable range. It awards points for shorter lengths, and penalizes longer lengths.
    """
    
    def __init__(self):
        pass
    
    def judge(self, prompts: List[str], completions: List[List[str]], **kwargs) -> List[float]:
        scores = []
        
        for prompt, completion_list in zip(prompts, completions):
            completion_scores = []
            
            for completion in completion_list:
                score = self._score_completion(prompt, completion)
                completion_scores.append(score)
            
            scores.extend(completion_scores)
        
        return scores
    
    def _score_completion(self, prompt: str, completion: str) -> float:
        """Score a single completion"""
        score = 0.0
        
        # Length scoring (0-1 points)
        prompt_length = len(prompt.split())
        completion_length = len(completion.split())
        difference = prompt_length - completion_length # we want positive for shorter completions
        
        # score as a percentage of the length improvement
        score += difference / prompt_length
        
        # normalize to -1 to 1 range
        # use tanh to squash the score to [-1, 1] range
        score = math.tanh(score)
        
        return score


class OpenAIJudge(BaseJudge):
    """
    Uses OpenAI's gpt 4.1 nano to assess whether the completion maintains the same meaning as the prompt.
    Also determines whether grammar and coherence are maintained.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.model = model
        
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
    
    def judge(self, prompts: List[str], completions: List[List[str]], **kwargs) -> List[float]:
        scores = []
        
        for prompt, completion_list in zip(prompts, completions):
            for completion in completion_list:
                try:
                    score = self._get_llm_score(prompt, completion)
                    scores.append(score)
                except Exception as e:
                    print(f"Error getting LLM score: {e}")
                    scores.append(0.0)  # Default score on error
        
        return scores
    
    def _get_llm_score(self, prompt: str, completion: str) -> float:
        """Get score from OpenAI's gpt 4.1 nano"""
        judge_prompt = self.judging_prompt.format(prompt=prompt, completion=completion)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=50, # example is only ~30 tokens
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
        except ValueError:
            return 0.0 # default score on error with parsing