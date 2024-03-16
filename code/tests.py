""" testing code """
import argparse
import metl
import torch
import utils


def load_checkpoint_run_inference(checkpoint_path, variants, dataset):
    """ loads a finetuned 3D model from a checkpoint and scores variants with the model """
    model, data_encoder = metl.get_from_checkpoint(checkpoint_path)

    # load the wild-type sequence and the PDB file (needed for 3D RPE) for the dataset
    datasets = utils.load_dataset_metadata()
    wt = datasets[dataset]["wt_aa"]
    pdb_fn = datasets[dataset]["pdb_fn"]

    variants = variants.split(";")

    encoded_variants = data_encoder.encode_variants(wt, variants)

    # set model to eval mode
    model.eval()

    # no need to compute gradients for inference
    with torch.no_grad():
        # note we are specifying the pdb_fn because this model uses 3D RPE
        predictions = model(torch.tensor(encoded_variants), pdb_fn=pdb_fn)

    print(predictions)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)

    parser.add_argument("--checkpoint_path",
                        help="path to saved METL target model",
                        type=str)
    parser.add_argument("--variants",
                        help="semicolon-separated list of variants to score",
                        type=str)
    parser.add_argument("--dataset",
                        help="protein dataset to load to obtain sequence and structure",
                        type=str)

    parsed_args = parser.parse_args()

    load_checkpoint_run_inference(args.checkpoint_path, args.variants, args.dataset)
