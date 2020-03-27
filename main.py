import argparse
import torch

from modules.utils import load_config, load_context, load_model, evaluate


def main(args):
    config = load_config(args.config_path)

    device = torch.device('cuda:1') if torch.cuda.is_available() \
        else torch.device('cpu')

    model = load_model(config, model_path=args.model_path)
    context = load_context(args.context_path)
    print('Context')
    print(context)
    print('='*50)

    while True:

        print('質問してください!')
        question = input()
        if question == 'q':
            break

        evaluate(
            question=question, context=context,
            model=model, config=config, device=device
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'context_path', type=str, help='Context file path'
    )
    parser.add_argument(
        '--model_path', type=str, help='Pretrained model path'
    )
    parser.add_argument(
        '--config_path', type=str, default='./models/config.yml',
        help='Config file path'
    )
    args = parser.parse_args()

    main(args)
