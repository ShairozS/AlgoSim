import json
import numpy as np
import pandas as pd
import re

class LLMOutput:
    '''
    A class that provides an interface to parse the output of multiple different LLM formulations
    into a consistent pattern. 

    Args:
        output (str): The natural langauge (decoded) output of the LLM, from the Huggingface generate() method for example
        template (str): What framework the LLM belongs to, currentely supports ['gpt', 'huggingface']
    '''

    def __init__(self, output, template = 'gpt'):
        self.template = template
        self.output = output
        self.samples = None # Populated when the parse_output() function is run

        
    def parse_output(self, verbose = 1):

        if self.output is not None:
            
            if self.template == 'gpt':
                try:
                    samples = [None]*len(self.output)
                except TypeError:
                    output = self.output.choices
                    samples = [None]*len(output)
                for i in range(len(samples)):
                    res = output[i]
                    res = res.to_dict()
                    res['function'] = self.extract_fun(res['message']['content'])
                    samples[i] = res

            elif self.template == 'huggingface':
                # ** NEEDS MORE TESTING **
                ## Output of tokenizer.batch_decode() method
                if self.output is not None:
                    output = self.output
                    samples = [{} for i in range(len(output))]
                    print(len(self.output), " programs found.")
                    
                    for i in range(len(samples)):

                        # Should be a response string
                        result_decoded = output[i]

                        # Empty response?
                        if len(result_decoded) == 0:
                            samples[i]['function'] = None
                            continue

                        # Split it on lines
                        result_decoded = result_decoded.split('\n')
                        #print(result_decoded)
                        result_decoded = [x for x in result_decoded if len(x) > 4]

                        ## ----- This section contains code for parsing and removing the predicted function from the LLM's output
                        ## Likely needs to be improved, but tested for several models
                        # Fine line containing answer section
                        start_answer = [k for k in range(len(result_decoded)) if 'Answer' in result_decoded[k] or '[CODE]' in result_decoded[k] or 'Solution' in result_decoded[k]]
                        
                        if len(start_answer) == 0:
                            # Try using end of instruction token instead
                            start_answer = [i for i in range(len(result_decoded)) if '[/INST]' in result_decoded[i]]
                            
                            # Finally, try using function definition token, may fail
                            if len(start_answer) == 0:
                                 start_answer = [i for i in range(len(result_decoded)) if 'def' in result_decoded[i]]
                            
                            try:
                                start_answer = max(start_answer)
                            except ValueError:
                                print("Could not extract function from program: ", i)
                                print("Skipping...")
                                samples[i]['function'] = None
                                continue
                        else:
                            start_answer = min(start_answer)
                        fn = result_decoded[start_answer:]

                        ## Extract function portion
                        try:
                            fn_start = min([k for k in range(len(fn)) if 'def' in fn[k]])
                            fn_end = max([k for k in range(len(fn)) if 'return' in fn[k]])
                            fn = fn[fn_start:fn_end+1]
                            samples[i]['function'] = '\n'.join(fn)
                        except ValueError:
                                print("Could not extract function from program: ", i)
                                print("Skipping...")
                                samples[i]['function'] = None
                                continue
                            
                else:
                    print("Template must be one of ['gpt', 'huggingface']")
                    assert False
                    
            self.samples = samples
            return(samples)

    @staticmethod
    def extract_fun(text):
        try:
            fun = text.split("def")[1].split('assert')[0].split("# Test")[0]
            fun = 'def' + fun
            fun = fun.split("```")[0]
        except Exception as e:
            fun = None
        return(fun)




class Problem:
    '''
    Define a problem statement and generate a sampling prompt appropriate for a particular
    LLM framework.

    Args:
        description (str): The natural langauge description of the programming problem
        test_cases (str): A Python codeblock of assert statements for the problem
        test_imports (str): Additional imports that may be required - NOT CURRENTELY IMPLEMENTED
    '''
    def __init__(self, description, test_cases, test_imports):

        self.description = description
        self.test_cases = test_cases
        self.test_imports = test_imports

    def get_prompt(self, template = 'openai'):

        if template == 'openai':
            with open('./Datasets/Prompts/prompt.txt') as f:
                base_prompt = f.read()
            base_prompt += '\n**Task**\n'
            base_prompt += self.description
            base_prompt += ' \n'
            base_prompt += "\nYour code should pass these tests:\n'''python"
            base_prompt += '\n' + '\n'.join(self.test_cases)
            base_prompt += "\n'''\n"
            return(base_prompt)

        elif template == 'huggingface':
            with open('./Datasets/Prompts/user_content_prompt.json') as f:
                prompt = json.load(f)
            newprompt = {"role": "user",
            "content": "Prompt: " + 'PROBLEM' + " Your code should pass the following test cases: \n " + "```python \n" +  'TESTS' + "\n ``` \n"
            }
            newprompt['content'] = newprompt['content'].replace('PROBLEM', self.description).replace('TESTS', '\n'.join(self.test_cases))
            prompt.append(newprompt)
            return(prompt)

