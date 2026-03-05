import numpy as np
import torch 
import re
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import PunktSentenceTokenizer
from context_cite import ContextCiter
from context_cite.context_partitioner import BaseContextPartitioner
from context_cite.solver import BaseSolver
from transformers.tokenization_utils_base import BatchEncoding, CharSpan
from transformers import PreTrainedTokenizer
from typing import Optional, Any, Dict, List
from numpy.typing import NDArray

#--- LongCite helper functions from query_longcite() method ---#
# helper functions for the query_longcite() method from LongCite-llama3.1-8b/modeling_llama.py file
# minimally modified

def text_split_by_punctuation(original_text, return_dict=False):
    # text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', original_text)  # separate period without space
    text = original_text
    custom_sent_tokenizer = PunktSentenceTokenizer()
    punctuations = r"([。；！？])"  # For Chinese support

    separated = custom_sent_tokenizer.tokenize(text)
    separated = sum([re.split(punctuations, s) for s in separated], [])
    # Put the punctuations back to the sentence
    for i in range(1, len(separated)):
        if re.match(punctuations, separated[i]):
            separated[i-1] += separated[i]
            separated[i] = ''

    separated = [s for s in separated if s != ""]
    if len(separated) == 1:
        separated = original_text.split('\n\n')
    separated = [s.strip() for s in separated if s.strip() != ""]
    if not return_dict:
        return separated
    else:
        pos = 0
        res = []
        for i, sent in enumerate(separated):
            st = original_text.find(sent, pos)
            assert st != -1, sent
            ed = st + len(sent)
            res.append(
                {
                    'c_idx': i,
                    'content': sent,
                    'start_idx': st,
                    'end_idx': ed,
                }
            )
            pos = ed
        return res

def get_prompt(context, question):
    sents = text_split_by_punctuation(context, return_dict=True)
    splited_context = ""
    splited_context_list = []
    separators = []
    for i, s in enumerate(sents):
        st, ed_old = s['start_idx'], s['end_idx']
        assert s['content'] == context[st:ed_old], s
        ed_new = sents[i+1]['start_idx'] if i < len(sents)-1 else len(context)
        sents[i] = {
            'content': context[st:ed_new],
            'start': st,
            'end': ed_new,
            'c_idx': s['c_idx'],
        }
        splited_context += f"<C{i}>"+context[st:ed_new]             # sent with marker with separator
        splited_context_list.append(context[st:ed_old])             # sent without marker without separator
        separators.append(context[ed_old:ed_new])                   # only separator
    prompt = '''Please answer the user's query based on the given document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's query.\n\n[Document Start]\n%s\n[Document End]\n\n[Query]\n%s\n\n[Remind]\nPlease answer the user's query based on the given document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's query.\n\n[Answer with citations]''' % (splited_context, question)
    return prompt, sents, splited_context, splited_context_list, separators

def get_citations(statement, sents):
    c_texts = re.findall(r'<cite>(.*?)</cite>', statement, re.DOTALL)
    spans = sum([re.findall(r"\[([0-9]+\-[0-9]+)\]", c_text, re.DOTALL) for c_text in c_texts], [])
    statement = re.sub(r'<cite>(.*?)</cite>', '', statement, flags=re.DOTALL)
    merged_citations = []
    for i, s in enumerate(spans):
        try:
            st, ed = [int(x) for x in s.split('-')]
            if st > len(sents) - 1 or ed < st:
                continue
            st, ed = max(0, st), min(ed, len(sents)-1)
            assert st <= ed, str(c_texts) + '\t' + str(len(sents))
            if len(merged_citations) > 0 and st == merged_citations[-1]['end_sentence_idx'] + 1:
                merged_citations[-1].update({
                    "end_sentence_idx": ed,
                    'end_char_idx': sents[ed]['end'],
                    'cite': ''.join([x['content'] for x in sents[merged_citations[-1]['start_sentence_idx']:ed+1]]),
                })
            else:
                merged_citations.append({
                    "start_sentence_idx": st,
                    "end_sentence_idx": ed,
                    "start_char_idx":  sents[st]['start'],
                    'end_char_idx': sents[ed]['end'],
                    'cite': ''.join([x['content'] for x in sents[st:ed+1]]),
                })
        except:
            print(c_texts, len(sents), statement)
            raise
    return statement, merged_citations[:3]

def postprocess(answer, sents, splited_context):
    res = []
    pos = 0
    new_answer = ""
    while True:
        st = answer.find("<statement>", pos)
        if st == -1:
            st = len(answer)
        ed = answer.find("</statement>", st)
        statement = answer[pos:st]
        if len(statement.strip()) > 5:
            res.append({
                "statement": statement,
                "citation": []
            })
            new_answer += f"<statement>{statement}<cite></cite></statement>"
        else:
            res.append({
                "statement": statement,
                "citation": None,
            })
            new_answer += statement
        
        if ed == -1:
            break

        statement = answer[st+len("<statement>"):ed]
        if len(statement.strip()) > 0:
            statement, citations = get_citations(statement, sents)
            res.append({
                "statement": statement,
                "citation": citations
            })
            c_str = ''.join(['[{}-{}]'.format(c['start_sentence_idx'], c['end_sentence_idx']) for c in citations])
            new_answer += f"<statement>{statement}<cite>{c_str}</cite></statement>"
        else:
            res.append({
                "statement": statement,
                "citation": None,
            })
            new_answer += statement
        pos = ed + len("</statement>")
    return {
        "answer": new_answer.strip(),
        "statements_with_citations": [x for x in res if x['citation'] is not None],
        "splited_context": splited_context.strip(),
        "all_statements": res,
    }

def truncate_from_middle(prompt, max_input_length=None, tokenizer=None):
    if max_input_length is None:
        return prompt
    else:
        assert tokenizer is not None
        tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokenized_prompt) > max_input_length:
            half = int(max_input_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        return prompt
    
#--- LongCite helper functions from query_longcite() method end ---#

#--- wrapper for implementation of token_to_chars() method for output_tokens ---#
class TokenToCharsWrapper:

    def __init__(self, token_ids: List[int], text: str, tokenizer: PreTrainedTokenizer):
        self.token_ids = token_ids 
        self.text = text 
        self.tokenizer = tokenizer 
        self.token_spans = self._find_token_spans()

    def _find_token_spans(self):
        """Find starting and ending character index for each token."""
        token_spans = []
        prev_end_idx = 0
        for token_id in self.token_ids:
            token_text = self.tokenizer.decode(token_id)
            start_idx = self.text.find(token_text, prev_end_idx)
            prev_end_idx = start_idx + len(token_text)
            token_spans.append((start_idx, prev_end_idx))
        return token_spans

    def token_to_chars(self, token_idx):
        if token_idx >= len(self.token_spans):
            return None 
        start_idx, end_idx = self.token_spans[token_idx]
        return CharSpan(start_idx, end_idx)

    
#--- subclass of ContextCiter and BaseContextPartitioner to work with LongCite-8B model and tokenizer ---#
class LongCiteContextPartitioner(BaseContextPartitioner):

    def __init__(self, context: str):
        super().__init__(context)
        self._cache = {}

    def split_context(self):
        """Split text into parts and cache the parts and separators."""
        _, _, _, parts, separators = get_prompt(context=self.context, question="")  # question is irrelevant here
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property 
    def parts(self):
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]
    
    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]
    
    @property
    def num_sources(self):
        return len(self.parts)
    
    def get_source(self, index: int):
        return self.parts[index]
    
    def get_context(self, mask: Optional[NDArray]=None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        parts = np.array(self.parts)[mask]
        context = ""
        for i, (part, sep) in enumerate(zip(parts, separators), start=1):
            context += part 
            if i < len(parts):
                context += sep
        return context

LONGCITE_GENERATE_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.95,
    "num_beams": 1,
    "top_p": 0.7,
}

LONGCITE_PROMPT_TEMPLATE = """Please answer the user's question based on the following document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's question.

[Document Start]
{context}
[Document End]

{query}
"""

class LongCiteContextCiter(ContextCiter):

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        context: str,
        query: str,
        source_type: str = "sentence",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[BaseSolver] = None,
        prompt_template: str = "",
        partitioner: Optional[BaseContextPartitioner] = None,
        max_input_length: int = 128000,
    ):
        super().__init__(model, tokenizer, context, query, source_type, 
                         generate_kwargs, num_ablations, ablation_keep_prob, 
                         batch_size, solver, prompt_template, partitioner)
        self.max_input_length = max_input_length
        self.generate_kwargs = generate_kwargs or LONGCITE_GENERATE_KWARGS

    def _get_prompt_ids(self, mask: Optional[NDArray]=None, return_prompt: bool=False):
        context = self.partitioner.get_context(mask)
        prompt, _, _, _, _ = get_prompt(context, self.query)
        prompt = truncate_from_middle(prompt, self.max_input_length, self.tokenizer)

        # apply chat template, get token_ids and get prompt string
        inputs = self.tokenizer.build_chat_input(prompt, history=[], role="user")
        del inputs["token_type_ids"]
        chat_prompt_ids = inputs["input_ids"][0].tolist()
        chat_prompt = self.tokenizer.decode(chat_prompt_ids)

        if return_prompt:
            return chat_prompt_ids, chat_prompt 
        else:
            return chat_prompt_ids 
        
    @property 
    def _output(self):
        if self._cache.get("output") is None:            
            prompt_ids, prompt = self._get_prompt_ids(return_prompt=True)
            prompt_ids = torch.tensor([prompt_ids], device=self.model.device)
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.get_command("<|user|>"), 
                            self.tokenizer.get_command("<|observation|>")] 
            with torch.inference_mode():
                outputs = self.model.generate(prompt_ids, **self.generate_kwargs, eos_token_id=eos_token_id)
            outputs_response = outputs.tolist()[0][prompt_ids.shape[1]:-1]  # cut until ending <|user|> tag
            response = self.tokenizer.decode(outputs_response)
            self._cache["output"] = prompt + response
        return self._cache["output"]
    
    @property 
    def _output_tokens(self):
        """Problem: LongCite tokenizer is Python based and does not support token_to_chars() method but
        the method is needed for the original logic in ContextCiter, therefore we use a wrapper for the
        output tokens.
        """
        output_ids = self.tokenizer.encode(self._output, add_special_tokens=False)
        wrapper = TokenToCharsWrapper(output_ids, self._output, self.tokenizer)
        output_tokens = BatchEncoding({"input_ids": output_ids})
        output_tokens.token_to_chars = wrapper.token_to_chars 
        return output_tokens 
    
    @property
    def response_dict(self):
        """Get LongCite results dictionary with info from post-processed model response."""
        if self._cache.get("response_dict") is None:
            _, sents, splited_context, _, _ = get_prompt(self.partitioner.get_context(), self.query)
            response_dict = postprocess(self.response.strip(), sents, splited_context)
            self._cache["response_dict"] = response_dict
        return self._cache["response_dict"]