import argparse
import itertools
import json
import logging
import pathlib
import pprint
import secrets

import skopt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tqdm

import data
import models

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(tqdm.tqdm)
handler.terminator = ""
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)


def get_parser(
    parser=argparse.ArgumentParser(description="run a definition modeling baseline"),
):
    parser.add_argument(
        "--do_htune",
        action="store_true",
        help="whether to perform hyperparameter tuning",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    # JZ --start--
    # parser.add_argument(
    #     "--source_arch",
    #     type=str,
    #     default="sgns",
    #     choices=("sgns", "char", "electra"),
    #     help="embedding architecture to use as source",
    # )

    parser.add_argument(
        "--source_arch",
        type=str,
        nargs="+",
        default=["sgns","char", "electra"],
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as source (can be one or more; e.g. '--source_arch sgns char electra')",
    )
    # JZ --end--
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / "defmod-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / "defmod-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--spm_model_path",
        type=pathlib.Path,
        default=None,
        help="use sentencepiece model, if required train and save it here",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("defmod-baseline-preds.json"),
        help="where to save predictions",
    )
    return parser


def get_search_space():
    """get hyperparmeters to optimize for"""
    search_space = [
        skopt.space.Real(1e-8, 1.0, "log-uniform", name="learning_rate"),
        skopt.space.Real(0.0, 1.0, "uniform", name="weight_decay"),
        skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_a"),
        skopt.space.Real(0.9, 1.0 - 1e-8, "log-uniform", name="beta_b"),
        skopt.space.Real(0.0, 0.9, "uniform", name="dropout"),
        skopt.space.Real(0.0, 1.0, "uniform", name="warmup_len"),
        skopt.space.Real(0.0, 1.0 - 1e-8, "uniform", name="label_smoothing"),
        skopt.space.Integer(1, 100, "log-uniform", name="batch_accum"),
        skopt.space.Integer(0, 5, "uniform", name="n_head_pow"),
        skopt.space.Integer(1, 6, "uniform", name="n_layers"),
    ]
    return search_space


def train(
    train_file,
    dev_file,
    source_arch=["sgns","char", "electra"], # JZ 
    summary_logdir=pathlib.Path("logs") / "defmod-htune",
    save_dir=pathlib.Path("models") / "defmod-baseline",
    device="cuda:0",
    spm_model_path=None,    
    epochs=100,
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    weight_decay=1e-6,
    patience=5,
    batch_accum=1,
    dropout=0.3,
    warmup_len=0.1,
    label_smoothing=0.1,
    n_head=4,
    n_layers=4,
):
    assert train_file is not None, "Missing dataset for training"
    assert dev_file is not None, "Missing dataset for development"



    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    # save_dir = save_dir / source_arch  # JZ
    save_dir = save_dir / "_".join(source_arch)  # JZ
    save_dir.mkdir(parents=True, exist_ok=True)
    ## make datasets
    train_dataset = data.get_train_dataset(train_file, spm_model_path, save_dir)
    dev_dataset = data.get_dev_dataset(
        dev_file, spm_model_path, save_dir, train_dataset
    )

    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."

    
    # JZ --start--
    # if source_arch == "electra":
    #     assert train_dataset.has_electra, "Training datatset contains no vector."
    # else:
    #     assert train_dataset.has_vecs, "Training datatset contains no vector."

    for arch in source_arch:
        if arch == "electra":
            assert train_dataset.has_electra, f"Training dataset is missing {arch} embeddings."
        else:
            assert train_dataset.has_vecs, f"Training dataset is missing {arch} embeddings."
    # JZ --end

    assert dev_dataset.has_gloss, "Development dataset contains no gloss."
    
    # JZ --start--
    # if source_arch == "electra":
    #     assert dev_dataset.has_electra, "Development dataset contains no vector."
    # else:
    #     assert dev_dataset.has_vecs, "Development dataset contains no vector."

    for arch in source_arch:
        if arch == "electra":
            assert dev_dataset.has_electra, f"Development dataset is missing {arch} embeddings."
        else:
            assert dev_dataset.has_vecs, f"Development dataset is missing {arch} embeddings."
    # JZ --end


    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset)
    dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False)
    ## make summary writer
    summary_writer = SummaryWriter(summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    logger.debug("Setting up training environment")

    model = models.DefmodModel(
        dev_dataset.vocab, n_head=n_head, n_layers=n_layers, dropout=dropout
    )

    model = model.to(device)
    model.train()

    # 3. declare optimizer & criterion
    ## Hyperparams

    # JZ 4 -start-->
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=learning_rate,
    #     betas=(beta1, beta2),
    #     weight_decay=weight_decay,
    # )

    # assign a larger lr so that the projected weights won't overpower the embedding weights
    # and ensure that weight decay won't interfere with embedding_weights
    embedding_weight_params = [p for name, p in model.named_parameters() if "embedding_weights" in name]
    other_params = [p for name, p in model.named_parameters() if "embedding_weights" not in name]
    embedding_wgt_lr = learning_rate * 50  # try 50, 100
    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},  
            {"params": embedding_weight_params, "lr": embedding_wgt_lr, "weight_decay": 0.0},  
        ],
        betas=(beta1, beta2),
    )
    # JZ 4 -end--<

    xent_criterion = nn.CrossEntropyLoss(ignore_index=model.padding_idx)
    if label_smoothing > 0.0:
        smooth_criterion = models.LabelSmoothingCrossEntropy(
            ignore_index=model.padding_idx, epsilon=label_smoothing
        )
    else:
        smooth_criterion = xent_criterion

    # vec_tensor_key = f"{source_arch}_tensor"  # JZ
    # vec_tensor_keys = ["sgns_tensor", "char_tensor", "electra_tensor"]  # JZ
    vec_tensor_keys = [f"{arch}_tensor" for arch in args.source_arch]  # JZ

    best_xent = float("inf")
    strikes = 0

    # 4. train model
    epochs_range = tqdm.trange(epochs, desc="Epochs")
    total_steps = (len(train_dataloader) * epochs) // batch_accum
    scheduler = models.get_schedule(
        optimizer, round(total_steps * warmup_len), total_steps
    )


    for epoch in epochs_range:
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        optimizer.zero_grad()

        # JZ -start-->
        # for i, batch in enumerate(train_dataloader):
        #     vec = batch[vec_tensor_key].to(device)
        #     gls = batch["gloss_tensor"].to(device)
        #     pred = model(vec, gls[:-1])
        for i, batch in enumerate(train_dataloader):
            vectors = {key.split("_")[0]: batch[key].to(device) for key in vec_tensor_keys if key in batch}
            gls = batch["gloss_tensor"].to(device)
            pred = model(vectors, gls[:-1])     
        # JZ -end--<

            loss = smooth_criterion(pred.view(-1, pred.size(-1)), gls.view(-1))

            # # JZ 3 -start-->
            # # Add L1 regularization to embedding weights
            # l1_lambda = 0.01  # moderate; May need to try a different value
            # l1_loss = l1_lambda * torch.norm(model.embedding_proj.weight, p=1)

            # # Combine losses
            # loss = loss + l1_loss
            # # JZ 3 -end--<

            loss.backward()

            # JZ tmp debugging -->
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name}")
            #         print(f"Gradient: {param.grad}")
            #     else:
            #         print(f"Parameter: {name} has no gradient")
            # print("Embedding Weights Grad:", model.embedding_weights.grad)  # JZ tmp            
            # for name, param in model.named_parameters():  # JZ tmp
            #     print(name, param.requires_grad)
            # JZ tmp --<
            
            grad_remains = True
            step = next(train_step)
            if i % batch_accum == 0:
                # if step % 3000 == 0: 
                #     print('-------------------Before step------------------')  # JZ tmp
                #     print("Embedding weights:", model.embedding_weights.detach().cpu().numpy())  # JZ tmp
                #     print("Projection layer weights:", model.embedding_proj.weight.detach().cpu().numpy())  # JZ tmp

                optimizer.step()
                if step % 4000 == 0: 
                    print(f'--------------After optimizer.step() at training step: {step}------------------')  # JZ tmp
                    print("Embedding weights:", model.embedding_weights.detach().cpu().numpy())  # JZ tmp
                    print("Projection layer weights:", model.embedding_proj.weight.detach().cpu().numpy())  # JZ tmp
                
                scheduler.step()
                optimizer.zero_grad()
                grad_remains = False
                summary_writer.add_scalar(
                    "defmod-train/lr", scheduler.get_last_lr()[0], step
                )
            # keep track of the train loss for this step
            with torch.no_grad():
                tokens = gls != model.padding_idx
                acc = (
                    ((pred.argmax(-1) == gls) & tokens).float().sum() / tokens.sum()
                ).item()
                xent_unsmoothed = xent_criterion(
                    pred.view(-1, pred.size(-1)), gls.view(-1)
                )
                summary_writer.add_scalar("defmod-train/xent_smooth", loss.item(), step)
                summary_writer.add_scalar("defmod-train/xent", xent_unsmoothed, step)
                summary_writer.add_scalar("defmod-train/acc", acc, step)
            # JZ -start-->
            # pbar.update(vec.size(0))                
            batch_size = list(vectors.values())[0].size(0)
            pbar.update(batch_size)
            # JZ -end--<
        if grad_remains:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        pbar.close()
        ## eval loop
        model.eval()
        with torch.no_grad():
            sum_dev_loss = 0.0
            sum_acc = 0
            ntoks = 0
            pbar = tqdm.tqdm(
                desc=f"Eval {epoch}",
                total=len(dev_dataset),
                disable=None,
                leave=False,
            )
            for batch in dev_dataloader:
                # vec = batch[vec_tensor_key].to(device)  # JZ
                vectors = {key.split("_")[0]: batch[key].to(device) for key in ["sgns_tensor", "char_tensor", "electra_tensor"] if key in batch}  # JZ
                gls = batch["gloss_tensor"].to(device)
                # pred = model(vec, gls[:-1])  # JZ
                pred = model(vectors, gls[:-1])  # JZ
                sum_dev_loss += F.cross_entropy(
                    pred.view(-1, pred.size(-1)),
                    gls.view(-1),
                    reduction="sum",
                    ignore_index=model.padding_idx,
                ).item()
                tokens = gls != model.padding_idx
                ntoks += tokens.sum().item()
                sum_acc += ((pred.argmax(-1) == gls) & tokens).sum().item()

                # JZ -start-->
                # pbar.update(vec.size(0))                
                batch_size = list(vectors.values())[0].size(0)
                pbar.update(batch_size)
                # JZ -end--<

            # keep track of the average loss & acc on dev set for this epoch
            new_xent = sum_dev_loss / ntoks
            summary_writer.add_scalar("defmod-dev/xent", new_xent, epoch)
            summary_writer.add_scalar("defmod-dev/acc", sum_acc / ntoks, epoch)
            pbar.close()
            if new_xent < (best_xent * 0.999):
                base_weights = torch.softmax(model.embedding_weights, dim=0)  # JZ 3
                projected_weights = model.embedding_proj(base_weights.unsqueeze(0)).squeeze(0)  # JZ 3
                logger.debug(
                    f"Epoch {epoch}, new best loss: {new_xent:.4f} < {best_xent:.4f}"
                    + f" (x 0.999 = {best_xent * 0.999:.4f})"
                    + f"\n Projected Embedding Weights: {projected_weights.cpu().numpy()}"  # JZ 3
                )
                best_xent = new_xent
                strikes = 0
            else:
                strikes += 1
            # check result if better
            if not (save_dir / "best_scores.txt").is_file():
                overall_best_xent = float("inf")
            else:
                with open(save_dir / "best_scores.txt", "r") as score_file:
                    overall_best_xent = float(score_file.read())
            # save result if better
            if new_xent < overall_best_xent:
                logger.debug(
                    f"Epoch {epoch}, new overall best loss: {new_xent:.4f} < {overall_best_xent:.4f}"
                )
                model.save(save_dir / "model.pt")
                with open(save_dir / "hparams.json", "w") as json_file:
                    # normalized_weights = torch.softmax(model.embedding_weights.detach().cpu(), dim=0)  # JZ 2
                    # Get softmax-normalized base weights
                    base_weights = torch.softmax(model.embedding_weights.detach().cpu(), dim=0)  # JZ 3
                    # instead of show the full 3d projection, only save the mean per row for each emb type as a reference
                    raw_projected_weights = model.embedding_proj.weight.detach().cpu().numpy().mean(axis=1).tolist() # JZ 4
                     
                    hparams = {
                        "learning_rate": learning_rate,
                        "beta1": beta1,
                        "beta2": beta2,
                        "weight_decay": weight_decay,
                        # "embedding_weights": {  # JZ 
                        #     source_arch[0]: model.embedding_weights.detach().cpu().tolist()[0],
                        #     source_arch[1]: model.embedding_weights.detach().cpu().tolist()[1],
                        #     source_arch[2]: model.embedding_weights.detach().cpu().tolist()[2],
                        # },
                        # "embedding_weights": {  # JZ 2 -Save normalized weights instead of raw values
                        #     source_arch[0]: normalized_weights[0].item(),
                        #     source_arch[1]: normalized_weights[1].item(),
                        #     source_arch[2]: normalized_weights[2].item(),
                        # },
                        "learned_softmax-normalized_embedding_weights": {  # JZ 3 ["sgns","char", "electra"]
                            source_arch[0]: base_weights[0].item(),
                            source_arch[1]: base_weights[1].item(),
                            source_arch[2]: base_weights[2].item(),
                        },
                        "learned_raw_projected_weights": { 
                            source_arch[0]: raw_projected_weights[0],
                            source_arch[1]: raw_projected_weights[1],
                            source_arch[2]: raw_projected_weights[2],
                        },
                    }
                    json.dump(hparams, json_file, indent=2)
                with open(save_dir / "best_scores.txt", "w") as score_file:
                    print(new_xent, file=score_file)

        if strikes >= patience:
            logger.debug("Stopping early.")
            epochs_range.close()
            break
        model.train()
    # return loss for gp minimize
    return best_xent


def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen, spm_model_name=args.spm_model_path
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1)
    model.eval()
    
    # vec_tensor_key = f"{args.source_arch}_tensor"  # JZ
    vec_tensor_keys = ["sgns_tensor", "char_tensor", "electra_tensor"]  # JZ

    if args.source_arch == "electra":
        assert test_dataset.has_electra, "File is not usable for the task"
    else:
        assert test_dataset.has_vecs, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset), disable=None)
        for batch in test_dataloader:
            # JZ -start-->
            # sequence = model.pred(batch[vec_tensor_key].to(args.device), decode_fn=test_dataset.decode, verbose=False)
            vectors = {key.split("_")[0]: batch[key].to(args.device) for key in vec_tensor_keys if key in batch}
            sequence = model.pred(vectors, decode_fn=test_dataset.decode, verbose=False)
            # JZ -end--<

            for id, gloss in zip(batch["id"], test_dataset.decode(sequence)):
                predictions.append({"id": id, "gloss": gloss})
            # pbar.update(batch[vec_tensor_key].size(0))  # JZ
            pbar.update(next(iter(vectors.values())).size(0))  # JZ

        pbar.close()
    # 3. dump predictions
    with open(args.pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    assert not (args.do_train and args.do_htune), "Conflicting options"
    if args.do_train:
        logger.debug("Performing defmod training")
        train(
            args.train_file,
            args.dev_file,
            args.source_arch,
            args.summary_logdir,
            args.save_dir,
            args.device,
            args.spm_model_path, #JZ
        )
    elif args.do_htune:
        logger.debug("Performing defmod hyperparameter tuning")
        search_space = get_search_space()

        source_arch_str = "_".join(args.source_arch)  # JZ

        @skopt.utils.use_named_args(search_space)
        def gp_train(**hparams):
            logger.debug(f"Hyperparams sampled:\n{pprint.pformat(hparams)}")
            best_loss = train(
                train_file=args.train_file,
                dev_file=args.dev_file,
                source_arch=args.source_arch,
                # summary_logdir=args.summary_logdir / args.source_arch / secrets.token_urlsafe(8),  # JZ
                summary_logdir=args.summary_logdir / source_arch_str / secrets.token_urlsafe(8),  # JZ
                save_dir=args.save_dir,
                device=args.device,
                spm_model_path=args.spm_model_path,
                learning_rate=hparams["learning_rate"],
                beta1=min(hparams["beta_a"], hparams["beta_b"]),
                beta2=max(hparams["beta_a"], hparams["beta_b"]),
                weight_decay=hparams["weight_decay"],
                batch_accum=hparams["batch_accum"],
                warmup_len=hparams["warmup_len"],
                label_smoothing=hparams["label_smoothing"],
                n_head=2 ** hparams["n_head_pow"],
                n_layers=hparams["n_layers"],
            )
            return best_loss

        result = skopt.gp_minimize(gp_train, search_space)
        # args.save_dir = args.save_dir / args.source_arch  # JZ
        args.save_dir = args.save_dir / source_arch_str  # JZ
        skopt.dump(result, args.save_dir / "results.pkl", store_objective=False)

    if args.do_pred:
        logger.debug("Performing defmod prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
