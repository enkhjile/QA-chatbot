import argparse
import torch

from torch.utils.data import DataLoader

from modules.tokenizers import MecabBertTokenizer, MecabBasicTokenizer
from modules.utils import to_list, load_config, load_context
from modules.modeling import ReformerQA
from modules.squad import (
    squad_convert_examples_to_features, compute_predictions_logits
)
from transformers.data.processors.squad import SquadExample, SquadResult


def main(args):
    print('='*50)
    print('Loading BOT...')
    config = load_config(args.config_path)

    device = torch.device('cuda:1') if torch.cuda.is_available() \
        else torch.device('cpu')

    model = ReformerQA(config)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    mecab_splitter = MecabBasicTokenizer(
        mecab_dict_path=args.mecab_dict_path
    )
    tokenizer = MecabBertTokenizer(
        args.vocab_path, max_len=config['max_seq_len'], do_basic_tokenize=False
    )

    context = load_context(args.context_path)
    context_text = ' '.join(mecab_splitter.tokenize(context))
    print('DONE!')
    print('='*50)
    print('CONTEXT:')
    print(context)
    print('='*50)
    while True:
        question = input('You>')
        if question == 'q':
            break
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
            max_seq_length=config['max_seq_len'],
            doc_stride=config['doc_stride'],
            max_query_length=config['max_query_length'],
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
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            do_lower_case=False,
            verbose_logging=False,
            version_2_with_negative=True,
            null_score_diff_threshold=0.0,
            tokenizer=tokenizer
        )

        answer = predictions['answer'].replace(' ', '')
        if not answer:
            answer = 'Answer can not be found from context!'
        print('BOT>', answer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'context_path', type=str, help='Context file path'
    )
    parser.add_argument(
        '--n_best_size', type=int,
        default=1
    )
    parser.add_argument(
        '--max_answer_length', type=int,
        default=30
    )
    parser.add_argument(
        '--mecab_dict_path', type=str,
        default='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd',
        help='Mecab dictionary path'
    )
    parser.add_argument(
        '--model_path', type=str, help='Pretrained model path'
    )
    parser.add_argument(
        '--config_path', type=str, default='./config.yml',
        help='Config file path'
    )
    parser.add_argument(
        '--vocab_path', type=str, default='./models/mecab_vocab.txt',
        help='Vocabulary file path')
    args = parser.parse_args()

    main(args)
