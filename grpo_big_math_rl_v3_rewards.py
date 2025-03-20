from math_verify import parse, verify
import re

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}")
    return [2.0 if verify(parse(r), parse(a)) else 0.0 for r, a in zip(responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>.*</reasoning>.+"
    responses = [completion[0]["content"] for completion in completions]
    print(f"\nResponse 1:")
    print(responses[0])
    print(f"Response 1 matches: {re.match(pattern, responses[0], re.DOTALL)}")
    print("-" * 40)
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [1 if match else 0.0 for match in matches]

import unittest



class TestCorrectnessRewardFunc(unittest.TestCase):
    
    def test_correct_answer_simple(self):
        # Test data
        prompts = [[{'role': 'system', 'content': 'system prompt'}, 
                   {'role': 'user', 'content': 'What is 2+2?'}]]
        completions = [[{'content': '<reasoning>\nI need to add 2 and 2.\n2 + 2 = 4\n</reasoning>\n\n4'}]]
        answer = ['4']
        
        # Capture stdout to verify print output
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0])
    
    def test_incorrect_answer_simple(self):
        # Test data
        prompts = [[{'role': 'system', 'content': 'system prompt'}, 
                   {'role': 'user', 'content': 'What is 2+3?'}]]
        completions = [[{'content': '<reasoning>\nI need to add 2 and 3.\n2 + 3 = 6\n</reasoning>\n\n6'}]]
        answer = ['5']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [0.0])
    
    def test_multiple_completions(self):
        # Test data
        prompts = [[{'role': 'user', 'content': 'Math problem'}]]
        completions = [
            [{'content': '4'}],
            [{'content': '5'}],
            [{'content': '6'}]
        ]
        answer = ['4', '4', '6']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0, 0.0, 2.0])
    
    def test_complex_math_problem(self):
        # Test data with more complex math
        prompts = [[{'role': 'user', 'content': 'Solve: 3x + 5 = 20'}]]
        completions = [[{'content': '<reasoning>\n3x + 5 = 20\n3x = 15\nx = 5\n</reasoning>\n\n5'}]]
        answer = ['5']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0])
    
    def test_equivalent_answers(self):
        # Test data with equivalent but differently formatted answers
        prompts = [[{'role': 'user', 'content': 'What is 1/2 + 1/4?'}]]
        completions = [[{'content': '<reasoning>\n1/2 + 1/4 = 2/4 + 1/4 = 3/4\n</reasoning>\n\n3/4'}]]
        answer = ['0.75']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Check if the verify function considers these equivalent
        # This test might pass or fail depending on how verify works with fractions vs decimals
        self.assertEqual(result, [2.0])
    
    def test_answer_extraction(self):
        # Test with reasoning and answer format
        prompts = [[{'role': 'user', 'content': 'What is the square root of 16?'}]]
        completions = [[{'content': '<reasoning>\nThe square root of 16 is 4 because 4 × 4 = 16.\n</reasoning>\n\n4'}]]
        answer = ['4']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0])
    
    def test_malformed_response(self):
        # Test with a response that doesn't follow the format
        prompts = [[{'role': 'user', 'content': 'What is 7×8?'}]]
        completions = [[{'content': 'I think the answer is 56.'}]]
        answer = ['56']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # This test depends on how parse handles malformed responses
        # It might pass if parse can extract "56" from the text
        self.assertEqual(result, [2.0])
    
    def test_empty_response(self):
        # Test with an empty response
        prompts = [[{'role': 'user', 'content': 'What is 10-3?'}]]
        completions = [[{'content': ''}]]
        answer = ['7']
        
        # Call the function and check it handles empty responses
        try:
            result = correctness_reward_func(prompts, completions, answer)
            self.assertEqual(result, [0.0])  # Expect 0 reward for empty response
        except Exception as e:
            # If parse can't handle empty strings, this might raise an exception
            self.fail(f"Empty response handling failed: {e}")
    
    def test_latex_expressions(self):
        # Test with LaTeX expressions in the response
        prompts = [[{'role': 'user', 'content': 'What is the derivative of x^2?'}]]
        completions = [[{'content': '<reasoning>\nTo find the derivative of x^2, I use the power rule: d/dx(x^n) = n*x^(n-1)\nFor x^2, n = 2, so d/dx(x^2) = 2*x^(2-1) = 2*x^1 = 2x\n</reasoning>\n\n2x'}]]
        answer = ['2x']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0])
        
        # Test with more complex LaTeX expressions
        prompts = [[{'role': 'user', 'content': 'Solve the equation: \\int_0^1 2x dx'}]]
        completions = [[{'content': '<reasoning>\n\\int_0^1 2x dx = x^2 |_0^1 = 1^2 - 0^2 = 1 - 0 = 1\n</reasoning>\n\n1'}]]
        answer = ['1']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0])
        
        # Test with fractions in LaTeX format
        prompts = [[{'role': 'user', 'content': 'Simplify \\frac{4}{8} + \\frac{1}{4}'}]]
        completions = [[{'content': '<reasoning>\n\\frac{4}{8} = \\frac{1}{2}\n\\frac{1}{2} + \\frac{1}{4} = \\frac{2}{4} + \\frac{1}{4} = \\frac{3}{4}\n</reasoning>\n\n\\frac{3}{4}'}]]
        answer = ['3/4']
        
        # Call the function
        result = correctness_reward_func(prompts, completions, answer)
        
        # Assertions
        self.assertEqual(result, [2.0])

class TestStrictFormatRewardFunc(unittest.TestCase):
    
    def test_correct_format(self):
        # Test with correctly formatted response
        completions = [[{'content': '<reasoning>\nThis is valid reasoning.\n</reasoning>\nThe answer is 42.'}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [1])
    
    def test_incorrect_format_no_tags(self):
        # Test with no reasoning tags
        completions = [[{'content': 'This has no reasoning tags. The answer is 42.'}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [0.0])
    
    def test_incorrect_format_wrong_order(self):
        # Test with tags in wrong order
        completions = [[{'content': '</reasoning>\nThis has reversed tags.\n<reasoning>'}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [0.0])
    
    def test_incorrect_format_not_at_start(self):
        # Test with reasoning tag not at the beginning
        completions = [[{'content': 'Some text before <reasoning>\nThis is invalid.\n</reasoning>'}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [0.0])
    
    def test_multiple_completions(self):
        # Test with multiple completions, some valid, some invalid
        completions = [
            [{'content': '<reasoning>\nValid format.\n</reasoning>\nAnswer: 10'}],
            [{'content': 'Invalid format without tags'}],
            [{'content': '<reasoning>\nInvalid because it does not contain any text after the end tag.\n</reasoning>'}]
        ]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [1, 0.0, 0.0])
    
    def test_empty_content(self):
        # Test with empty content
        completions = [[{'content': ''}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [0.0])
    
    def test_multiline_reasoning(self):
        # Test with multi-line reasoning
        completions = [[{'content': '<reasoning>\nLine 1\nLine 2\nLine 3\n</reasoning>\nAnswer: 42'}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [1])
    
    def test_case_sensitivity(self):
        # Test case sensitivity of tags
        completions = [[{'content': '<REASONING>\nThis has uppercase tags.\n</REASONING>'}]]
        result = strict_format_reward_func(completions)
        self.assertEqual(result, [0.0])

class TestCorrectnessRewardFunc(unittest.TestCase):
    def test_correct_answer(self):
        # Test when the extracted answer matches the expected answer
        prompts = [[{'content': 'What is 2+2?'}]]
        completions = [[{'content': '<reasoning>\nTo find 2+2, I add the numbers.\n2+2=4\n</reasoning>\nAnswer: 4'}]]
        answer = ['4']
        result = correctness_reward_func(prompts, completions, answer)
        self.assertEqual(result, [2.0])
    
    def test_incorrect_answer(self):
        # Test when the extracted answer doesn't match the expected answer
        prompts = [[{'content': 'What is 2+2?'}]]
        completions = [[{'content': '<reasoning>\nI think 2+2=5\n</reasoning>\nAnswer: 5'}]]
        answer = ['4']
        result = correctness_reward_func(prompts, completions, answer)
        self.assertEqual(result, [0.0])
    
    def test_multiple_answers(self):
        # Test with multiple completions, some correct, some incorrect
        prompts = [[{'content': 'What is 2+2?'}], [{'content': 'What is 3+3?'}]]
        completions = [
            [{'content': '<reasoning>\n2+2=4\n</reasoning>\nAnswer: 4'}],
            [{'content': '<reasoning>\n3+3=7\n</reasoning>\nAnswer: 7'}]
        ]
        answer = ['4', '6']
        result = correctness_reward_func(prompts, completions, answer)
        self.assertEqual(result, [2.0, 0.0])
    
    def test_complex_math_answer(self):
        # Test with more complex mathematical expressions
        prompts = [[{'content': 'Solve x^2 = 9'}]]
        completions = [[{'content': '<reasoning>\nx^2 = 9\nx = ±3\n</reasoning>\nAnswer: x = 3 or x = -3'}]]
        answer = ['x = 3 or x = -3']
        result = correctness_reward_func(prompts, completions, answer)
        self.assertEqual(result, [2.0])

if __name__ == '__main__':
    unittest.main()