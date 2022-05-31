from pathlib import Path
from tqdm import tqdm

from plugins.PluginLoader import PluginLoader
from lib.cli import FullPaths
from lib.utils import get_folder

class ExportObject(object):
    filename = ''
    def __init__(self, subparser, command, description='default'):
        self.create_parser(subparser,command,description)

    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Convert a source image to a new one with the face swapped.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
        self.add_optional_arguments(self.parser)
        self.parser.set_defaults(func=self.process_arguments)

    def add_optional_arguments(self, parser):
        parser.add_argument('-m', '--model-dir',
                            action=FullPaths,
                            dest="model_dir",
                            default="models",
                            help="Model directory. A directory containing the trained model \
                    you wish to process. Defaults to 'models'")

        parser.add_argument('-t', '--trainer',
                            type=str,
                            choices=("Original", "LowMem", "GAN"), # case sensitive because this is used to load a plug-in.
                            default="Original",
                            help="Select the trainer that was used to create the model.")


        parser.add_argument('-s', '--swap-model',
                            action="store_true",
                            dest="swap_model",
                            default=False,
                            help="Swap the model. Instead of A -> B, swap B -> A.")

        parser.add_argument('-e', '--export_path',
                            default='model_export',
                            help="Swap the model. Instead of A -> B, swap B -> A.")

        return parser
    
    def process_arguments(self, arguments):
        self.arguments = arguments
        print("Model Directory: {}".format(self.arguments.model_dir))
        self.process()

    def process(self):
        # Original & LowMem models go with Adjust or Masked converter
        # GAN converter & model must go together
        # Note: GAN prediction outputs a mask + an image, while other predicts only an image
        model_name = self.arguments.trainer

        model = PluginLoader.get_model(model_name)(get_folder(self.arguments.model_dir))
        if not model.load(self.arguments.swap_model):
            print('Model Not Found! A valid model must be provided to continue!')
            exit(1)

        model.export(self.arguments.export_path)