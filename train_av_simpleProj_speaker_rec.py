#!/usr/bin/env python3
import os
import sys
import pickle as pkl
import time

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
Recipe for training speaker recognition models on the MavCeleb dataset using a simple projection head and a combined audio-visual classifier.
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
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])


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

        # === ===
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

    def test_ER_with_output(self, 
            test_set_audio, 
            test_set_visual, 
            trials_fn,
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

        # Load Trials-File
        if trials_fn is not None:
            trials_df = pd.read_csv(trials_fn)
            if not all(col in trials_df.columns for col in ["label", "voice_id", "face_id"]):
                raise ValueError("Trials CSV must contain columns: label, voice_id, face_id")
        

        # Compute embeddings for all test utterances
        self.modules.projection_model_audio.eval()
        self.modules.projection_model_visual.eval()
        # Generate mapping: id -> wav
        embeddings_audio = {}
        scores_audio = {}
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

                # L2-normalization for embeddings
                audio_embedding = F.normalize(audio_embedding, p=2, dim=1)

                with torch.no_grad():
                    embs = self.modules.projection_model_audio(audio_embedding)
                    embs_classified = self.modules.classifier(embs)

                for i in range(len(batch.id)):
                    uid = batch.id[i]
                    embeddings_audio[uid] = embs[i].squeeze().cpu().numpy()
                    scores_audio[uid] = embs_classified[i].squeeze().cpu().numpy()

        embeddings_visual = {}
        scores_visual = {}
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

                # L2-normalization for embeddings
                image_embedding = F.normalize(image_embedding, p=2, dim=1)

                with torch.no_grad():
                    embs = self.modules.projection_model_visual(image_embedding)
                    embs_classified = self.modules.classifier(embs)

                for i in range(len(batch.id)):
                    uid = batch.id[i]
                    embeddings_visual[uid] = embs[i].squeeze().cpu().numpy()
                    scores_visual[uid] = embs_classified[i].squeeze().cpu().numpy()

        # === Calculate similarities from trials ===
        results_list = None
        accuracy_embs = 0.0
        accuracy_scores = 0.0
        if trials_fn is not None:
            scores, labels = [], []
            label_score_list = []
            correct_pred_embs = 0
            correct_pred_scores = 0
            all_preds = 0
            cos_sim_av_list = []
            results_list = list()
            for _, row in tqdm(trials_df.iterrows(), total=len(trials_df), desc="Evaluating pairs"):
                voice_id, face_id, label = row["voice_id"], row["face_id"], row["label"]
                # if voice_id in embeddings_audio and face_id in embeddings_visual:
                emb1, emb2 = embeddings_audio[voice_id], embeddings_visual[face_id]
                score1, score2 = scores_audio[voice_id], scores_visual[face_id]
                embs = emb1 + emb2
                cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                cos_sim_av_list.append(cos_sim)
                # print(f"Cosine similarity between audio and visual embeddings for pair ({voice_id}, {face_id}): {cos_sim:.4f}")
                embs = embs / np.linalg.norm(embs)
                score_from_combined_embs = self.modules.classifier(torch.tensor(embs).to(self.device).unsqueeze(0)).detach().squeeze().cpu().numpy()
                scores = score1 + score2

                argmax_score_from_combined_embs = torch.argmax(torch.tensor(score_from_combined_embs))
                argmax_scores = torch.argmax(torch.tensor(scores))

                try:
                    label_ind = hparams["speaker_encoder"].encode_label_torch(label)
                    if argmax_score_from_combined_embs == label_ind:
                        correct_pred_embs += 1
                    if argmax_scores == label_ind:
                        correct_pred_scores += 1
                    all_preds += 1
                except KeyError:
                    continue
                
                results_list.append({
                        'voice_id': voice_id,
                        'face_id': face_id,
                        'label': label,
                        'label_ind': label_ind.item(),
                        'predicted_label_from_combined_embs': argmax_score_from_combined_embs.item(),
                        'predicted_label_from_scores': argmax_scores.item(),
                    })

            accuracy_embs = correct_pred_embs / all_preds
            accuracy_scores = correct_pred_scores / all_preds
            print(f"Accuracy from combined embeddings: {accuracy_embs:.4f} | Accuracy from scores: {accuracy_scores:.4f}", flush=True)
            print(f"Cos-sim statistics - mean {np.mean(cos_sim_av_list):.4f} - min {np.min(cos_sim_av_list):.4f} - max {np.max(cos_sim_av_list):.4f} - std {np.std(cos_sim_av_list):.4f}", flush=True)

        out_dict = {
            'audio_embeddings': embeddings_audio,
            'visual_embeddings': embeddings_visual,
            'results_list': results_list,
        }

        return {
                "accuracy_embs": accuracy_embs,
                "error_rate_embs": 1 - accuracy_embs,
                "accuracy_scores": accuracy_scores,
                "error_rate_scores": 1 - accuracy_scores,
                'audio_embeddings': embeddings_audio,
                'visual_embeddings': embeddings_visual,
                'results_list': results_list,
        }

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
        emb_VGG_fn = os.path.join(hparams['visual_embedding_folder'], img.replace('.jpg', '.pkl'))
        emb_ag_fn = os.path.join(hparams['visual_embedding_VIT_folder'], img.replace('.jpg', '.pkl'))
        # Load and normalize ECAPA embedding
        with open(emb_VGG_fn, "rb") as f:
            embedding_VGG = pkl.load(f)
        embedding_VGG = torch.nn.functional.normalize(embedding_VGG, p=2, dim=0)
        # Load and normalize Age-Gender embedding
        with open(emb_ag_fn, "rb") as f:
            embedding_ag = pkl.load(f)
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
    
    datasets['train_unheard_audio'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"audio_train_unheard_csv"],
            dynamic_items=[embedding_audio_pipeline, label_pipeline],
            output_keys=["id", "embedding_audio", "speaker", "speaker_encoded"],
        )
    
    datasets['train_unheard_visual'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"visual_train_unheard_csv"],
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
            dynamic_items=[embedding_image_pipeline],
            output_keys=["id", "embedding_image"],
        )
    
    datasets[f'val_unheard_audio'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"val_unheard_audio_csv"],
            dynamic_items=[embedding_audio_pipeline],
            output_keys=["id", "embedding_audio"],
        )
    
    datasets[f'val_unheard_visual'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"val_unheard_visual_csv"],
            dynamic_items=[embedding_image_pipeline],
            output_keys=["id", "embedding_image"],
        )

    # Test sets
    datasets[f'test_heard_audio'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"test_heard_audio_csv"],
            dynamic_items=[embedding_audio_pipeline],
            output_keys=["id", "embedding_audio"],
        )
    
    datasets[f'test_heard_visual'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"test_heard_visual_csv"],
            dynamic_items=[embedding_image_pipeline],
            output_keys=["id", "embedding_image"],
        )
    
    datasets[f'test_unheard_audio'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"test_unheard_audio_csv"],
            dynamic_items=[embedding_audio_pipeline],
            output_keys=["id", "embedding_audio"],
        )
    
    datasets[f'test_unheard_visual'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"test_unheard_visual_csv"],
            dynamic_items=[embedding_image_pipeline],
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

    lid_brain.checkpointer.recover_if_possible(min_key="loss")  # oder "error", je nach dem, was du beim Speichern nutzt
    epoch = lid_brain.hparams.epoch_counter.current
    print("Loaded Checkpoint from epoch:", epoch, flush=True)
    lid_brain.hparams.train_logger.log_stats(
        {"Loaded Epoch": epoch, "Phase": "TEST"}
    )

    ############################################################################
    # Calculate scores and EER for validation sets
    ############################################################################

    print("Final evaluation on test set:", flush=True)

    # Test ER with save results

    # Train data
    print('Extract embeddings for heard train data ...', flush=True)
    heard_train_results_dict = lid_brain.test_ER_with_output( 
                test_set_audio=lid_brain.hparams.datasets["train_audio"],
                test_set_visual=lid_brain.hparams.datasets["train_visual"],
                trials_fn=None,
                audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
                visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options
        )
    out_fn = os.path.join(lid_brain.hparams.output_folder, "heard_train_results.pkl")
    print(f"Heard train results saved to: {out_fn}", flush=True)
    with open(out_fn, "wb") as f:
        pkl.dump(heard_train_results_dict, f)

    print('Extract embeddings for unheard train data ...', flush=True)
    unheard_train_results_dict = lid_brain.test_ER_with_output( 
                test_set_audio=lid_brain.hparams.datasets["train_unheard_audio"],
                test_set_visual=lid_brain.hparams.datasets["train_unheard_visual"],
                trials_fn=None,
                audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
                visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options
        )
    out_fn = os.path.join(lid_brain.hparams.output_folder, "unheard_train_results.pkl")
    print(f"Unheard train results saved to: {out_fn}", flush=True)
    with open(out_fn, "wb") as f:
        pkl.dump(unheard_train_results_dict, f)


    # Val data
    print('Extract embeddings for heard val data ...', flush=True)
    heard_val_results_dict = lid_brain.test_ER_with_output( 
                test_set_audio=lid_brain.hparams.datasets["val_heard_audio"],
                test_set_visual=lid_brain.hparams.datasets["val_heard_visual"],
                trials_fn=None,
                audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
                visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options
        )
    out_fn = os.path.join(lid_brain.hparams.output_folder, "heard_val_results.pkl")
    print(f"Heard val results saved to: {out_fn}", flush=True)
    with open(out_fn, "wb") as f:
        pkl.dump(heard_val_results_dict, f)

    print('Extract embeddings for unheard val data ...', flush=True)
    unheard_val_results_dict = lid_brain.test_ER_with_output( 
                test_set_audio=lid_brain.hparams.datasets["val_unheard_audio"],
                test_set_visual=lid_brain.hparams.datasets["val_unheard_visual"],
                trials_fn=None,
                audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
                visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options
        )
    out_fn = os.path.join(lid_brain.hparams.output_folder, "unheard_val_results.pkl")
    print(f"Unheard val results saved to: {out_fn}", flush=True)
    with open(out_fn, "wb") as f:
        pkl.dump(unheard_val_results_dict, f)


    # Test data
    print("Testing on 'heard' dataset with output...", flush=True)
    heard_test_results_dict = lid_brain.test_ER_with_output( 
                test_set_audio=lid_brain.hparams.datasets["test_heard_audio"],
                test_set_visual=lid_brain.hparams.datasets["test_heard_visual"],
                trials_fn=lid_brain.hparams.test_heard_trials_csv,
                audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
                visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options
        )
    out_fn = os.path.join(lid_brain.hparams.output_folder, "heard_test_results.pkl")
    print(f"Heard test results saved to: {out_fn}", flush=True)
    with open(out_fn, "wb") as f:
        pkl.dump(heard_test_results_dict, f)
    print(f"Heard test results saved to: {out_fn}", flush=True)

    print("Testing on 'unheard' dataset with output...", flush=True)
    unheard_test_results_dict = lid_brain.test_ER_with_output( 
                test_set_audio=lid_brain.hparams.datasets["test_unheard_audio"],
                test_set_visual=lid_brain.hparams.datasets["test_unheard_visual"],
                trials_fn=lid_brain.hparams.test_unheard_trials_csv,
                audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
                visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options
        )
    out_fn = os.path.join(lid_brain.hparams.output_folder, "unheard_test_results.pkl")
    print(f"Unheard test results saved to: {out_fn}", flush=True)
    with open(out_fn, "wb") as f:
        pkl.dump(unheard_test_results_dict, f)
    print(f"Unheard test results saved to: {out_fn}", flush=True)


    # Test auf 'heard' Dataset
    heard_result_av = lid_brain.test_ER(
        test_set_audio=lid_brain.hparams.datasets["test_heard_audio"],
        test_set_visual=lid_brain.hparams.datasets["test_heard_visual"],
        trials_fn=lid_brain.hparams.test_heard_trials_csv,
        scores_fn="",
        audio_only=False,
        audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
        visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options,
    )
    print(heard_result_av)

    heard_result_a = lid_brain.test_ER(
        test_set_audio=lid_brain.hparams.datasets["test_heard_audio"],
        test_set_visual=lid_brain.hparams.datasets["test_heard_visual"],
        trials_fn=lid_brain.hparams.test_heard_trials_csv,
        scores_fn="",
        audio_only=True,
        audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
        visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options,
    )
    print(heard_result_a)

    unheard_result_av = lid_brain.test_ER(
        test_set_audio=lid_brain.hparams.datasets["test_unheard_audio"],
        test_set_visual=lid_brain.hparams.datasets["test_unheard_visual"],
        trials_fn=lid_brain.hparams.test_unheard_trials_csv,
        scores_fn="",
        audio_only=False,
        audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
        visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options,
    )
    print(unheard_result_av)

    unheard_result_a = lid_brain.test_ER(
        test_set_audio=lid_brain.hparams.datasets["test_unheard_audio"],
        test_set_visual=lid_brain.hparams.datasets["test_unheard_visual"],
        trials_fn=lid_brain.hparams.test_unheard_trials_csv,
        scores_fn="",
        audio_only=True,
        audio_test_loader_kwargs=lid_brain.hparams.audio_test_dataloader_options,
        visual_test_loader_kwargs=lid_brain.hparams.visual_test_dataloader_options,
    )
    print(unheard_result_a)


    # Logge die Ergebnisse
    lid_brain.hparams.train_logger.log_stats(
        {"Epoch": 'Final', "Phase": "TEST"},
        test_stats={
            "heard_av_er": heard_result_av["error_rate"],
            "heard_a_er": heard_result_a["error_rate"],
            "unheard_av_er": unheard_result_av["error_rate"],
            "unheard_a_er": unheard_result_a["error_rate"],
        },
    )
    lid_brain.hparams.train_logger.log_stats(
        {"Epoch": 'Final', "Phase": "TEST"},
        test_stats={
            "heard_av_acc": f'{heard_result_av["accuracy"]:.4f}',
            "heard_a_acc": f'{heard_result_a["accuracy"]:.4f}',
            "unheard_av_acc": f'{unheard_result_av["accuracy"]:.4f}',
            "unheard_a_acc": f'{unheard_result_a["accuracy"]:.4f}',
        },
    )



    # Test ER with save results




    print('Done.')
    