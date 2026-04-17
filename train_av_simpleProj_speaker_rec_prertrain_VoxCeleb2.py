#!/usr/bin/env python3
import os
import sys
import pickle as pkl
import time
import random

import torchaudio
import torchvision.io as io
from torchvision import transforms
import torch.nn.functional as F
import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.logger import get_logger

from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader

import warnings
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")

"""
Recipe for pre-training speaker recognition models on the VoxCeleb2 dataset using a simple projection head and a combined audio-visual classifier.
"""

logger = get_logger(__name__)


# Brain class for Language ID training
class LID(sb.Brain):
    


    def prepare_features_visual(self, imgs, stage):
        imgs, lens = imgs
        return imgs, lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : torch.Tensor
            torch.Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats_audio, lens_audio = self.prepare_feature_audio(batch.sig, stage)
        embeddings_audio = self.modules.embedding_model_audio(feats_audio, lens_audio)
        outputs_audio = self.modules.classifier(embeddings_audio)


        imgs, _ = self.prepare_features_visual(batch.img_tensor, stage)
        embeddings_visual = self.modules.embedding_model_visual(imgs)
        outputs_visual = self.modules.classifier(embeddings_visual)

        return outputs_audio, lens_audio

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        targets = batch.speaker_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            # if targets.shape[0] != predictions.shape[0]:
            #     if hasattr(self.hparams, "wav_augment"):
            #         targets = self.hparams.wav_augment.replicate_labels(targets)
            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        loss = self.hparams.compute_cost(predictions, targets)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                # "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            
            # Save the current checkpoint and delete previous checkpoints,
            #self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])
            # self.checkpointer.save_checkpoint(meta=stats)
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])
            # self.checkpointer.save_checkpoint(meta=stats)


        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def test_ER(self, 
            test_set_audio, 
            test_set_visual, 
            trials_fn, 
            scores_fn,
            calculate_EER=True,
            audio_only=False,
            min_key="error", 
            audio_test_loader_kwargs=None, 
            visual_test_loader_kwargs=None,
            progressbar=None):
        """Evaluate Equal Error Rate (EER) on unseen speakers using a trials file.

        Arguments
        ---------
        test_set : DynamicItemDataset
            SpeechBrain dataset containing test samples.
        trials_fn : str
            Path to a CSV file containing 'voice_id', 'face_id', and 'label' columns.
            Labels must be 1 for same-speaker, 0 for different-speaker.
        min_key : str
            Metric key used for loading the best model (default = 'error').
        test_loader_kwargs : dict
            Loader options for test data.

        Returns
        -------
        dict : {
            "eer": float,
            "threshold": float,
            "n_trials": int
        }
        """

        import pandas as pd
        import numpy as np
        from sklearn.metrics import roc_curve
        from tqdm import tqdm
        import torch

        from torch.utils.data import DataLoader
        from speechbrain.dataio.dataloader import LoopedLoader

        stage = sb.Stage.TEST

        if progressbar is None:
            progressbar = not self.noprogressbar

        torch.manual_seed(hparams["seed"])
        torch.cuda.manual_seed_all(hparams["seed"])

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        if not ( isinstance(test_set_audio, DataLoader) or isinstance(test_set_audio, LoopedLoader)):
            audio_test_loader_kwargs["ckpt_prefix"] = None
            test_set_audio = self.make_dataloader(test_set_audio, stage, **audio_test_loader_kwargs)
        
        if not ( isinstance(test_set_visual, DataLoader) or isinstance(test_set_visual, LoopedLoader)):
            visual_test_loader_kwargs["ckpt_prefix"] = None
            test_set_visual = self.make_dataloader(test_set_visual, stage, **visual_test_loader_kwargs)

        self.modules.eval()

        print(f"\n🔍 Computing EER using trials file: {trials_fn}")

        # Lade Trials-Datei
        trials_df = pd.read_csv(trials_fn)
        if not all(col in trials_df.columns for col in ["label", "voice_id", "face_id"]):
            raise ValueError("Trials CSV must contain columns: label, voice_id, face_id")
        

        # === Embeddings für alle Test-Utterances berechnen ===
        self.modules.projection_model_audio.eval()
        self.modules.projection_model_visual.eval()
        # Erzeuge Mapping: id -> wav
        embeddings_audio = {}
        with torch.no_grad():
            for batch in tqdm(
                test_set_audio,
                dynamic_ncols=True,
                disable= True, #not enable,
                colour=self.tqdm_barcolor["test"],
                ):

                batch = batch.to(self.device)

                # Compute features, embeddings and output
                audio_embedding, lens = batch.embedding_audio

                # L2-Normalisierung der Embeddings
                audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

                with torch.no_grad():
                    embs = self.modules.projection_model_audio(audio_embedding)
                    embs_classified = self.modules.classifier(embs)

                for i in range(len(batch.id)):
                    uid = batch.id[i]
                    embeddings_audio[uid] = embs_classified[i].squeeze().cpu().numpy()

        embeddings_visual = {}
        with torch.no_grad():
            for batch in tqdm(
                test_set_visual,
                dynamic_ncols=True,
                disable=True,
                colour=self.tqdm_barcolor["test"],
                ):

                batch = batch.to(self.device)

                # Compute features, embeddings and output
                image_embedding, lens = batch.embedding_image

                # L2-Normalisierung der Embeddings
                image_embedding = F.normalize(image_embedding, p=2, dim=1)

                with torch.no_grad():
                    embs = self.modules.projection_model_visual(image_embedding)
                    embs_classified = self.modules.classifier(embs)

                for i in range(len(batch.id)):
                    uid = batch.id[i]
                    embeddings_visual[uid] = embs_classified[i].squeeze().cpu().numpy()

        # === Ähnlichkeiten aus Trials berechnen ===
        scores, labels = [], []
        label_score_list = []
        correct_pred = 0
        all_preds = 0
        for _, row in tqdm(trials_df.iterrows(), total=len(trials_df), desc="Evaluating pairs"):
            voice_id, face_id, label = row["voice_id"], row["face_id"], row["label"]
            # if voice_id in embeddings_audio and face_id in embeddings_visual:
            emb1, emb2 = embeddings_audio[voice_id], embeddings_visual[face_id]
            if audio_only:
                scores = emb1
            else:
                scores = emb1 + emb2
            try:
                label_ind = hparams["speaker_encoder"].encode_label_torch(label)
                if torch.argmax(torch.tensor(scores)) == label_ind:
                    correct_pred += 1
                all_preds += 1
            except KeyError:
                continue

        accuracy = correct_pred / all_preds

        return {
                "accuracy": accuracy,
                "error_rate": 1 - accuracy,}

    def get_random_orthogonal_vector(self, inp_vectors):
        """
        Erzeugt für jeden Eingabevektor (batchweise) einen Zufallsvektor,
        der orthogonal zu diesem steht.
        """
        # inp_vectors: [batch, dim]
        v = F.normalize(inp_vectors, p=2, dim=1)        # Referenzvektoren
        r = torch.randn_like(v)                         # Zufallsvektoren

        # Projektion von r auf v entfernen → orthogonale Komponente
        proj = torch.sum(r * v, dim=1, keepdim=True) * v
        v_perp = F.normalize(r - proj, p=2, dim=1)

        # # Kontrolle (sollte ≈ 0 sein)
        # dot = torch.sum(v * v_perp, dim=1)
        # print("Orthogonalitätsfehler (sollte ≈ 0 sein):", dot.mean().item())

        return v_perp

    def fit_multimodal(
            self,
            epoch_counter,
            audio_train_loader,
            visual_train_loader,
            audio_train_loader_kwargs={},
            visual_train_loader_kwargs={},
            audio_valid_loader_kwargs={},
            visual_valid_loader_kwargs={},
        ):
        """Custom multimodal training loop with two dataloaders."""

        if self.test_only:
            logger.info(
                "Test only mode, skipping training and validation stages."
            )
            return
        

        if not ( isinstance(audio_train_loader, DataLoader) or isinstance(audio_train_loader, LoopedLoader) ):
            audio_train_loader = self.make_dataloader( audio_train_loader, stage=sb.Stage.TRAIN, **audio_train_loader_kwargs)
        if not ( isinstance(visual_train_loader, DataLoader) or isinstance(visual_train_loader, LoopedLoader) ):
            visual_train_loader = self.make_dataloader( visual_train_loader, stage=sb.Stage.TRAIN, **visual_train_loader_kwargs)

        self.on_fit_start()

        for epoch in epoch_counter:
            self.on_stage_start(sb.Stage.TRAIN, epoch)
            self.modules.train()
            avg_loss = 0.0
            steps = 0

            print(f"\n--- Starting epoch {epoch} ---", flush=True)

            # Visual-Loader kann kürzer sein → cycle
            min_iterations = min(len(audio_train_loader), len(visual_train_loader))
            log_frequ = int(min_iterations/20)
            print(f"Number of training iterations for this epoch: {min_iterations}", flush=True)

            t_start = time.time()
            for audio_batch, visual_batch in zip(audio_train_loader, visual_train_loader):
                audio_batch = audio_batch.to(self.device)
                visual_batch = visual_batch.to(self.device)

                audio_embedding, len_a = audio_batch.embedding_audio
                image_embedding, len_v = visual_batch.embedding_image

                # L2-Normalisierung der Embeddings
                # audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
                # image_embedding = F.normalize(image_embedding, p=2, dim=1)

                if self.hparams.do_embedding_augmentation:

                    # Zufallsorthogonale Vektoren erzeugen
                    audio_aug_vec_lens = torch.abs(torch.randn(audio_embedding.shape[0]) * self.hparams.emb_aug_std + self.hparams.emb_aug_mean).to(self.device)
                    audio_aug_vec = self.get_random_orthogonal_vector(audio_embedding)
                    audio_aug_vec = audio_aug_vec * audio_aug_vec_lens.unsqueeze(1)
                    audio_embedding = audio_embedding + audio_aug_vec

                    image_aug_vec_lens = torch.abs(torch.randn(image_embedding.shape[0]) * self.hparams.emb_aug_std + self.hparams.emb_aug_mean).to(self.device)
                    image_aug_vec = self.get_random_orthogonal_vector(image_embedding)
                    image_aug_vec = image_aug_vec * image_aug_vec_lens.unsqueeze(1)
                    image_embedding = image_embedding + image_aug_vec

                    # audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
                    # image_embedding = F.normalize(image_embedding, p=2, dim=1)

                    # # Audio embedding augmentation
                    # audio_aug_vec_lens = torch.abs(torch.randn(audio_embedding.shape[0]) * self.hparams.emb_aug_std).to(self.device)
                    # audio_aug_vec = torch.rand_like(audio_embedding)-0.5
                    # audio_aug_vec = F.normalize(audio_aug_vec, p=2, dim=1)
                    # audio_aug_vec = audio_aug_vec * audio_aug_vec_lens.unsqueeze(1)
                    # audio_embedding = audio_embedding + audio_aug_vec
                    # audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
                    # 
                    # # Image embedding augmentation
                    # image_aug_vec_lens = torch.abs(torch.randn(image_embedding.shape[0]) * self.hparams.emb_aug_std).to(self.device)
                    # image_aug_vec = torch.rand_like(image_embedding)-0.5
                    # image_aug_vec = F.normalize(image_aug_vec, p=2, dim=1)
                    # image_aug_vec = image_aug_vec * image_aug_vec_lens.unsqueeze(1)
                    # image_embedding = image_embedding + image_aug_vec
                    # image_embedding = F.normalize(image_embedding, p=2, dim=1)


                # === Audio forward ===
                proj_audio = self.modules.projection_model_audio(audio_embedding)
                out_a = self.modules.classifier(proj_audio)
                loss_a = self.compute_objectives((out_a, len_a), audio_batch, sb.Stage.TRAIN)

                # === Image forward ===
                proj_image = self.modules.projection_model_visual(image_embedding)
                out_v = self.modules.classifier(proj_image)
                loss_v = self.compute_objectives((out_v, len_v), visual_batch, sb.Stage.TRAIN)

                # === Combine losses ===
                loss = (loss_a + loss_v) / 2.0

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.detach()
                steps += 1

                if steps % log_frequ == 0:
                    ratio = 100 * steps / min_iterations # in Prozent
                    avg_loss_ = avg_loss.item() / steps
                    duration = time.time() - t_start
                    remaining = (min_iterations - steps) * (duration / steps)
                    print(f"Epoch: {epoch} | Ratio: {ratio:.1f}% | Step: {steps} of {min_iterations} | Loss: {avg_loss_:.4f} | Elapsed: {duration:.1f}s | Remaining: {remaining:.1f}s", flush=True)

            avg_loss /= steps
            self.on_stage_end(sb.Stage.TRAIN, avg_loss, epoch)

            # Validation
            self.evaluate(
                epoch=epoch,
            )

    def evaluate(
            self,
            epoch,
            stage=sb.Stage.VALID,
        ):
        """Evaluation loop for a single dataloader."""

        self.modules.eval()
        self.on_stage_start(stage)

        # Test auf 'heard' Dataset
        heard_result_AV = self.test_ER(
            test_set_audio=self.hparams.datasets["val_heard_audio"],
            test_set_visual=self.hparams.datasets["val_heard_visual"],
            trials_fn=self.hparams.val_heard_trials_csv,
            scores_fn="",
            audio_only=False,
            audio_test_loader_kwargs=self.hparams.audio_test_dataloader_options,
            visual_test_loader_kwargs=self.hparams.visual_test_dataloader_options,
        )

        heard_result_A = self.test_ER(
            test_set_audio=self.hparams.datasets["val_heard_audio"],
            test_set_visual=self.hparams.datasets["val_heard_visual"],
            trials_fn=self.hparams.val_heard_trials_csv,
            scores_fn="",
            audio_only=True,
            audio_test_loader_kwargs=self.hparams.audio_test_dataloader_options,
            visual_test_loader_kwargs=self.hparams.visual_test_dataloader_options,
        )



        # Logge die Ergebnisse
        self.hparams.train_logger.log_stats(
            {"Epoch": epoch, "Phase": "TEST"},
            test_stats={
                "heard_er": (heard_result_AV["error_rate"] + heard_result_A["error_rate"]) / 2,
            },
        )

        # avg_loss = (heard_result["eer"] + unheard_result["eer"])/2
        avg_loss = (heard_result_AV["error_rate"] + heard_result_A["error_rate"]) / 2

        self.on_stage_end(stage, avg_loss, epoch)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_language` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'lang01': 0, 'lang02': 1, ..)
    speaker_encoder = sb.dataio.encoder.CategoricalEncoder()


    ################################################
    # Audio and image embedding pipelines
    ################################################

    # train_audio_ECAPA_embedding_folder = hparams["audio_embedding_ECAPA_folder"]
    # audio_embedding_W2V2_folder = hparams["audio_embedding_W2V2_folder"]
    @sb.utils.data_pipeline.takes("id", "wav")
    @sb.utils.data_pipeline.provides("id", "embedding_audio")
    def embedding_audio_pipeline(ID, wav):
        emb_ecapa_fn = os.path.join(hparams['audio_embedding_ECAPA_folder'], wav.replace('.wav', '.pkl'))
        emb_ag_fn = os.path.join(hparams['audio_embedding_AG_folder'], wav.replace('.wav', '.pkl'))
        # Load and normalize ECAPA embedding
        with open(emb_ecapa_fn, "rb") as f:
            embedding_ecapa = pkl.load(f)
        embedding_ecapa = torch.nn.functional.normalize(embedding_ecapa, p=2, dim=0)
        # Load and normalize Age-Gender embedding
        with open(emb_ag_fn, "rb") as f:
            embedding_ag = pkl.load(f)
        embedding_ag = torch.nn.functional.normalize(embedding_ag, p=2, dim=0)
        # Concatenate embeddings
        embedding = torch.cat( (embedding_ecapa, embedding_ag), dim=0 )
        return ID, embedding
    
    

    @sb.utils.data_pipeline.takes("id", "img")
    @sb.utils.data_pipeline.provides("id", "embedding_image")
    def embedding_image_pipeline(ID, img):
        emb_VGG_fn = os.path.join(hparams['visual_embedding_folder'], img.replace('.jpg', '_embeddings.pkl'))
        emb_ag_fn = os.path.join(hparams['visual_embedding_VIT_folder'], img.replace('.jpg', '_embeddings.pkl'))
        # Load and normalize ECAPA embedding
        with open(emb_VGG_fn, "rb") as f:
            embedding_VGG = pkl.load(f)
            idx = random.randint(0, embedding_VGG.shape[0]-1)
            embedding_VGG = embedding_VGG[idx]
        embedding_VGG = torch.nn.functional.normalize(embedding_VGG, p=2, dim=0)
        # Load and normalize Age-Gender embedding
        with open(emb_ag_fn, "rb") as f:
            embedding_ag = pkl.load(f)
            idx = random.randint(0, embedding_ag.shape[0]-1)
            embedding_ag = embedding_ag[idx]
        embedding_ag = torch.nn.functional.normalize(embedding_ag, p=2, dim=0)
        # Concatenate embeddings
        embedding = torch.cat( (embedding_VGG, embedding_ag), dim=0 )
        return ID, embedding
    

    @sb.utils.data_pipeline.takes("id", "img")
    @sb.utils.data_pipeline.provides("id", "embedding_image")
    def embedding_image_pipeline_eval(ID, img):
        emb_VGG_fn = os.path.join(hparams['visual_embedding_folder'], img.replace('.jpg', '_embeddings.pkl'))
        emb_ag_fn = os.path.join(hparams['visual_embedding_VIT_folder'], img.replace('.jpg', '_embeddings.pkl'))
        # Load and normalize ECAPA embedding
        with open(emb_VGG_fn, "rb") as f:
            embedding_VGG = pkl.load(f)
            idx = int(embedding_VGG.shape[0]/2) # always take the middle one for evaluation
            embedding_VGG = embedding_VGG[idx]
        embedding_VGG = torch.nn.functional.normalize(embedding_VGG, p=2, dim=0)
        # Load and normalize Age-Gender embedding
        with open(emb_ag_fn, "rb") as f:
            embedding_ag = pkl.load(f)
            idx = int(embedding_ag.shape[0]/2) # always take the middle one for evaluation
            embedding_ag = embedding_ag[idx]
        embedding_ag = torch.nn.functional.normalize(embedding_ag, p=2, dim=0)
        # Concatenate embeddings
        embedding = torch.cat( (embedding_VGG, embedding_ag), dim=0 )
        return ID, embedding


    ################################################
    # Label data processing pipeline
    ################################################

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("speaker")
    @sb.utils.data_pipeline.provides("speaker", "speaker_encoded")
    def label_pipeline(speaker):
        yield speaker
        speaker_encoded = speaker_encoder.encode_label_torch(speaker)
        yield speaker_encoded

    # Create dataset objects "train", "valid", and "test"
    datasets = {}

    # Training sets
    datasets[f'train_audio'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"audio_train_csv"],
            dynamic_items=[embedding_audio_pipeline, label_pipeline],
            output_keys=["id", "embedding_audio", "speaker", "speaker_encoded"],
        )
    
    datasets[f'train_visual'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"visual_train_csv"],
            dynamic_items=[embedding_image_pipeline, label_pipeline],
            output_keys=["id", "embedding_image", "speaker", "speaker_encoded"],
        )

    # Validation sets
    datasets[f'val_heard_audio'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"val_heard_audio_csv"],
            dynamic_items=[embedding_audio_pipeline],
            output_keys=["id", "embedding_audio"],
        )
    
    datasets[f'val_heard_visual'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"val_heard_visual_csv"],
            dynamic_items=[embedding_image_pipeline_eval],
            output_keys=["id", "embedding_image"],
        )


    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    os.makedirs(hparams["save_folder"], exist_ok=True)
    speaker_encoder_file = os.path.join(
        hparams["save_folder"], "speaker_encoder.txt"
    )
    speaker_encoder.load_or_create(
        path=speaker_encoder_file,
        from_didatasets=[datasets["train_audio"]],
        output_key="speaker",
    )

    speaker_encoder.expect_len(hparams["n_speaker"])

    return datasets, speaker_encoder


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed_all(hparams["seed"])

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # # Data preparation, to be run on only one process.
    # sb.utils.distributed.run_on_main(
    #     prepare_common_language,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_folder": hparams["save_folder"],
    #         "skip_prep": hparams["skip_prep"],
    #     },
    # )

    # Create dataset objects "train", "dev", and "test" and speaker_encoder
    datasets, speaker_encoder = dataio_prep(hparams)
    hparams["speaker_encoder"] = speaker_encoder
    hparams["datasets"] = datasets

    # Fetch and load pretrained modules
    if "pretrainer" in hparams:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

        if 'skip_keys' in hparams:
            state_dict = hparams["embedding_model"].state_dict()
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")["model"]
            # Entferne inkompatible Keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc.")}
            state_dict.update(pretrained_dict)
            hparams["embedding_model"].load_state_dict(state_dict, strict=False)

    # Initialize the Brain object to prepare for mask training.


    lid_brain = LID(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if 'test_only' in hparams:
        lid_brain.test_only = hparams['test_only']

    # print(f'lid_brain.test_only: {lid_brain.test_only}', flush=True)
    # print(lasjkd)

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    # lid_brain.fit(
    #     epoch_counter=lid_brain.hparams.epoch_counter,
    #     train_set=datasets["train"],
    #     valid_set=datasets["dev"],
    #     train_loader_kwargs=hparams["train_dataloader_options"],
    #     valid_loader_kwargs=hparams["test_dataloader_options"],
    # )
    lid_brain.fit_multimodal(
        epoch_counter=lid_brain.hparams.epoch_counter,
        audio_train_loader=datasets["train_audio"],
        visual_train_loader=datasets["train_visual"],
        audio_train_loader_kwargs=hparams["audio_train_dataloader_options"],
        visual_train_loader_kwargs=hparams["visual_train_dataloader_options"],
        audio_valid_loader_kwargs=hparams["audio_valid_dataloader_options"],
        visual_valid_loader_kwargs=hparams["visual_valid_dataloader_options"],
    )
    


    print('Done.')
    