import random
import numpy as np
import torch
import logging

from seqprotes import seqprotes

class STEPSPrompt:
    def __init__(self, model, tokenizer, token_embedding, args, device):
        """Initialize STEPSPrompt
        
        Args:
            model: CLIP model
            tokenizer: Tokenizer
            token_embedding: Token embedding layer
            args: Configuration parameters
            device: Running device
        """
        self.model = model
        self.tokenizer = tokenizer
        self.token_embedding = token_embedding
        self.args = args
        self.device = device
        
        # Save the best result
        self.best_sim = -1000 * args.loss_weight
        self.best_text = ""
        
        # seqprotes related state
        self.top_10_I_matrix = None
        self.G_hat = None
        self.previous_I_y = None
        self.top_n = args.top_n
        
        # Initialize input
        self.input_ids = None
        self.top_indices_new = None
    
    def step(self, target_features):
        """Execute single step attack
        
        Args:
            target_features: Target feature vector
            
        Returns:
            new_input_ids: New input token ids
            max_loss_value: Loss value for current step
        """
        # Calculate gradients and get top indices
        grad, _ = self._token_gradients(self.input_ids, target_features)
        top_indices_gradient = (-grad).topk(self.args.topk, dim=1).indices
        
        # Update top indices
        top_indices = (torch.cat((self.top_indices_new, top_indices_gradient), dim=1) 
                      if self.top_10_I_matrix is not None 
                      else top_indices_gradient)
        
        # Create loss function and run seqprotes optimization
        loss_fun = self._create_loss_fun(top_indices, target_features)
        i_opt, y_opt, seqprotes_info = seqprotes(
            loss_fun, 
            top_indices.shape[0],
            top_indices.shape[1],
            self.args.sample_bs,
            log=True,
            k_top=10,
            is_max=True,
            r=self.args.rank,
            with_info_full=True,
            previous_I_y=self.previous_I_y,
            G_hat=self.G_hat
        )
        
        # Update input ids
        new_input_ids = top_indices[torch.arange(top_indices.shape[0]), np.array(i_opt)]
        
        # Update seqprotes state
        self._update_seqprotes_state(seqprotes_info, top_indices)
        
        # Update input
        self.input_ids = new_input_ids.unsqueeze(1)
        
        return new_input_ids, y_opt.item()
    
    def run(self, target_features):
        """Run complete prompt tuning process
        
        Args:
            target_features: Target feature vector
            
        Returns:
            best_text: Best adversarial prompt text
        """
        # Initialize input
        self.input_ids = torch.randint(
            0, 40408, 
            (self.args.prompt_len, self.args.batch_size)
        ).to(self.device)
        
        # Process target feature batch
        curr_target_features = self._get_batch_features(target_features)
        
        # Add tracking variable before for loop
        last_loss_value = None
        unchanged_count = 0
        max_unchanged = 30  # Maximum allowed unchanged times
        
        for step_i in range(self.args.iter):
            # Execute single step attack
            new_input_ids, max_loss_value = self.step(curr_target_features)
            
            # if self.args.update_topn:
                # self.top_n = max(32 - step_i, 4)
                # min(step_i + 1, 64)
                # max(32 - step_i, 8)   # l(t) = max(32-t, 4)
            # Color print
            # print(f"\033[94mtop_n: {self.top_n}\033[0m")
            # Check if loss has changed
            if last_loss_value is not None and abs(max_loss_value - last_loss_value) < 1e-6:
                unchanged_count += 1
                if unchanged_count >= max_unchanged:
                    logging.info(f"Loss value unchanged for {max_unchanged} iterations, early stopping at step {step_i}")
                    break
            else:
                unchanged_count = 0
            
            last_loss_value = max_loss_value
            
            # Decode and evaluate results
            new_text = self._decode_ids(new_input_ids)
            
            # Update best result
            if max_loss_value > self.best_sim:
                self.best_sim = max_loss_value
                self.best_text = new_text
                self.best_step = step_i
                logging.info(f"best_text: {self.best_text}")
                logging.info(f"best_sim: {self.best_sim}")
            
            logging.info(f"step: {step_i}, cosine sim: {max_loss_value}, prompt: {new_text}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
        return self.best_text, self.best_sim, self.best_step

    def _token_gradients(self, input_ids, target_features):
        """Calculate token gradients"""
        embed_weights = self.model.token_embedding.weight
        input_len = input_ids.shape[0]
        
        if input_len > 75:
            input_ids = input_ids[:75]
            
        # Create one-hot vector
        one_hot = torch.zeros(
            input_ids.shape[0],
            embed_weights.shape[0],
            device=self.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            input_ids,
            torch.ones(one_hot.shape[1], 1, device=self.device, dtype=embed_weights.dtype)
        )    
        one_hot.requires_grad_()

        # Process padding
        padding_len = 77 - input_len
        padding_ids = torch.zeros(padding_len, 1, dtype=input_ids.dtype, device=self.device)
        padding_ids[0, 0] = 49406
        padding_ids[1, 0] = 49407
        
        # Build complete input
        dummy_ids = torch.full((input_len, 1), -1, dtype=input_ids.dtype, device=self.device)
        padding_ids_end = padding_ids[1:, :]
        if padding_ids_end.dim() == 1:
            padding_ids_end = padding_ids_end.unsqueeze(0)
        padding_dummy_ids = torch.cat(
            [padding_ids[0,:].unsqueeze(0), dummy_ids, padding_ids_end], 
            dim=0
        ).to(self.device)

        # Create embedding
        one_hot_tmp = torch.zeros(
            padding_ids.shape[0], 
            embed_weights.shape[0], 
            device=self.device, 
            dtype=embed_weights.dtype
        )
        one_hot_tmp.scatter_(
            1, 
            padding_ids, 
            torch.ones(one_hot_tmp.shape[1], 1, device=self.device, dtype=embed_weights.dtype)
        )
        
        start_embeds = (one_hot_tmp[0,:] @ embed_weights).unsqueeze(0)
        end_embeds = (one_hot_tmp[1:,:] @ embed_weights)
        if end_embeds.dim() == 1:
            end_embeds = end_embeds.unsqueeze(0)

        input_embeds = (one_hot @ embed_weights)
        
        # Forward propagation
        full_embeds = torch.cat(
            [start_embeds, input_embeds, end_embeds], 
            dim=0
        ).unsqueeze(0)
        
        logits_per_image, _ = self.model.forward_text_embedding(
            full_embeds, 
            padding_dummy_ids.t(), 
            target_features
        )
        loss = 1 - logits_per_image.mean()
        
        # Backward propagation
        loss.backward()
        grad = one_hot.grad.clone()
        
        return grad, padding_dummy_ids.t()

    def _create_loss_fun(self, top_indices, target_features):
        """Create loss function"""
        def index_top_indices(I):
            I = np.array(I)
            batch_size = I.shape[0]
            row_indices = np.arange(self.input_ids.shape[0]).reshape(1, self.input_ids.shape[0]).repeat(batch_size, axis=0)
            result = top_indices[row_indices, I]
            return result

        def loss_fun(I):
            batch_ids = index_top_indices(I)
            batch_cosim_scores = self._batch_ids_scores(batch_ids, target_features)
            loss_np = batch_cosim_scores.detach().cpu().numpy().squeeze()
            return loss_np
        
        return loss_fun

    def _batch_ids_scores(self, batch_token_ids, target_features):
        """Calculate cosine similarity scores for batch"""
        seq_len = batch_token_ids.shape[1]
        expanded_token_ids = torch.zeros(
            batch_token_ids.shape[0], 
            77, 
            dtype=batch_token_ids.dtype, 
            device=self.device
        )
        expanded_token_ids[:, 0] = 49406
        expanded_token_ids[:, 1:seq_len+1] = batch_token_ids
        expanded_token_ids[:, seq_len+1] = 49407

        with torch.no_grad():
            batch_text_features = self.model.encode_text(expanded_token_ids)

        batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
        batch_cosim_scores = batch_text_features @ target_features.t()
        return batch_cosim_scores

    def _update_seqprotes_state(self, seqprotes_info, top_indices):
        """Update seqprotes related state"""
        if self.top_n > 0:
            top_10_elements = self._find_top_n_elements(
                seqprotes_info['y_list'],
                top_n=self.top_n,
                per_vector_top_n=None
            )
            top_10_I_list = self._extract_rows_from_I_list(seqprotes_info['I_list'], top_10_elements)
            self.top_10_I_matrix = np.column_stack(top_10_I_list)
            self.previous_I_y = np.transpose(self.top_10_I_matrix)
            top_10_I_matrix_torch = torch.from_numpy(self.top_10_I_matrix).to(self.device)
            self.G_hat = self._extract_tensors(seqprotes_info['P_list'][-1], self.top_10_I_matrix)
            self.top_indices_new = top_indices[
                torch.arange(top_indices.shape[0]).unsqueeze(1),
                top_10_I_matrix_torch
            ]
        else:
            # When top_n=0, clear all states
            self.top_10_I_matrix = None
            self.previous_I_y = None
            self.G_hat = None
            self.top_indices_new = None

    def _get_batch_features(self, target_features):
        """Get batch target features"""
        if self.args.batch_size is None:
            curr_features = target_features
        else:
            curr_indx = torch.randperm(len(target_features))[:self.args.batch_size]
            curr_features = target_features[curr_indx]
        curr_features /= curr_features.norm(dim=-1, keepdim=True)
        return curr_features

    def _decode_ids(self, input_ids):
        """Decode token ids to text"""
        return self.tokenizer.decode(input_ids.cpu().numpy())

    @staticmethod
    def _find_top_n_elements(vectors_list, top_n=10, per_vector_top_n=None):
        """Find top n elements from the list of vectors"""
        if per_vector_top_n is None:
            per_vector_top_n = len(vectors_list[0])

        all_elements = []
        for i, vector in enumerate(vectors_list):
            if len(vector) > per_vector_top_n:
                top_indices = np.argsort(vector)[-per_vector_top_n:]
            else:
                top_indices = np.argsort(vector)
            for idx in top_indices:
                all_elements.append((vector[idx], i, idx))
        return sorted(all_elements, key=lambda x: x[0], reverse=True)[:top_n]

    @staticmethod
    def _extract_rows_from_I_list(I_list, top_elements):
        """Extract rows from I_list"""
        return [I_list[list_idx][vector_idx, :] for _, list_idx, vector_idx in top_elements]

    @staticmethod
    def _extract_tensors(G_list, Index_matrix):
        """Extract tensors"""
        G1, G2, G3 = G_list
        m = Index_matrix.shape[0]
        
        G1_hat = G1[:, Index_matrix[0], :]
        
        G2_hat_list = []
        for i in range(1, m - 1):
            G2_i = G2[i - 1:i, :, Index_matrix[i], :]
            G2_hat_list.append(G2_i)
        G2_hat = np.concatenate(G2_hat_list, axis=0)
        G3_hat = G3[:, Index_matrix[-1], :]
        
        return [G1_hat, G2_hat, G3_hat]