import yaml
import torch

from torch.utils.data import DataLoader
from modules.modeling import ReformerQA
from modules.tokenizers import MecabBertTokenizer, MecabBasicTokenizer
from transformers.data.processors.squad import SquadExample, SquadResult
from modules.squad import (
    squad_convert_examples_to_features, compute_predictions_logits
)


def load_config(config_path):
    with open(config_path, 'r') as ymlfile:
        config = yaml.full_load(ymlfile)
    return config


def load_model(config, model_path):
    print('='*50)
    print('Loading BOT...')
    model = ReformerQA(config['model'])
    if model_path:
        model.load_state_dict(torch.load(model_path))

    print('DONE!')
    print('='*50)

    return model


def load_context(context_path):
    with open(context_path, 'r') as reader:
        lines = reader.readlines()
    context = ''.join(lines)
    return context


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(question, context, model, config, device=None):
    print('YOU> ', question)
    mecab_splitter = MecabBasicTokenizer(
        mecab_dict_path=config['tokenizer']['mecab_dict']
    )
    tokenizer = MecabBertTokenizer(
        config['tokenizer']['vocab'], max_len=config['tokenizer']['max_len'],
        do_basic_tokenize=False
    )
    context_text = ' '.join(mecab_splitter.tokenize(context))
    question_text = ' '.join(mecab_splitter.tokenize(question))

    example = SquadExample(
        qas_id='answer',
        question_text=question_text,
        context_text=context_text,
        answer_text=None,
        start_position_character=None,
        title=None
    )

    features, dataset = squad_convert_examples_to_features(
        [example],
        tokenizer,
        max_seq_length=config['model']['max_seq_len'],
        doc_stride=config['predict']['doc_stride'],
        max_query_length=config['predict']['max_query_length'],
        is_training=False,
        )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_results = []
    model.to(device)
    model.eval()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            "input_ids": batch[0]
        }
        with torch.no_grad():
            outputs = model(**inputs)
        example_indices = batch[3]

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    predictions = compute_predictions_logits(
        [example],
        features,
        all_results,
        n_best_size=config['predict']['n_best_size'],
        max_answer_length=config['predict']['max_answer_length'],
        do_lower_case=False,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer
    )

    answer = predictions['answer'].replace(' ', '')
    if not answer:
        answer = 'Answer can not be found from context!'
    print('BOT> ', answer)
