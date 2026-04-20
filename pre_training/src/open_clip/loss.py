import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
TEXT_CLS_TAG = 0
TEXT_TOKEN_TAG = 1
TEXT_MASK_TAG = 2
TEXT_SPAN_NUMS_TAG = 4
TEXT_REPEATED_VECTOR_TAG = 5


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None, tag=0):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
        tag=tag,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
        tag=tag,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None, tag=0):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
        tag=tag,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
        tag=tag,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
        tag=tag,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
        tag=tag,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tag, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        ctx.tag = tag
        return neighbour_exchange(from_rank, to_rank, tensor, group=group, tag=tag)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, ctx.tag, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None, tag=0):
    return NeighbourExchange.apply(from_rank, to_rank, group, tag, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tag, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        ctx.tag = tag
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group, tag=tag)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, ctx.tag, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None, tag=0):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tag, tensor_to_left, tensor_to_right)


class SigLipConceptLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def extract_and_pool_embeddings(self, embeddings, span_positions, span_nums, repeated_vector):
        """
        Extracts embeddings for specified spans and performs mean pooling.

        Parameters:
        - embeddings (torch.Tensor): Tensor of shape [batch_size, sequence_length, embedding_size].
        - span_positions (torch.Tensor): Tensor of shape [batch_size, concept_max_num, 2], where 2 is the start and end positions of the span.

        Returns:
        - pooled_embeddings (torch.Tensor): Tensor of shape [batch_size, num_spans, embedding_size].
        """
        span_positions = span_positions + 1  # We add 1 to the start and end positions to account for the [CLS] token
        batch_size, sequence_length, embedding_size = embeddings.shape
        padded_span_mask = span_positions.sum(dim=-1).bool()
        spans_per_batch = span_nums
        # # Get batch indices and span start/end indices
        batch_indices = repeated_vector

        padded_span_mask_x, padded_span_mask_y = torch.nonzero(padded_span_mask, as_tuple=True)
        start_indices, end_indices = span_positions[padded_span_mask_x, padded_span_mask_y].unbind(dim=-1)
        # Create a tensor for the spans
        max_num_spans = span_positions.size(1)
        max_span_length = (end_indices - start_indices).max().item()

        # Initialize the pooled embeddings tensor and mask
        pooled_embeddings = torch.zeros(batch_size, max_num_spans, embedding_size, device=embeddings.device)

        # Create indices for gathering
        span_range = torch.arange(max_span_length, device=embeddings.device).unsqueeze(0)  # Shape: [1, max_span_length]

        # Prepare the indices for advanced indexing
        batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, max_span_length)
        start_indices_expanded = start_indices.unsqueeze(1).expand(-1, max_span_length)
        end_indices_expanded = end_indices.unsqueeze(1).expand(-1, max_span_length)

        # Calculate the actual indices
        actual_indices = start_indices_expanded + span_range
        # Create a mask for valid indices
        valid_mask = actual_indices < end_indices_expanded

        # Clamp the actual indices to avoid out-of-bounds
        actual_indices = torch.clamp(actual_indices, min=0, max=sequence_length - 1)  # Ensure no out-of-bounds

        # Gather embeddings for each span
        gathered_embeddings = embeddings[batch_indices_expanded, actual_indices]

        # Zero out invalid indices
        gathered_embeddings[~valid_mask] = 0  # Set invalid indices to zero

        # Perform mean pooling, ignoring zeros
        span_means = gathered_embeddings.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).float()

        # Create indices for batch and span positions
        span_indices_range = torch.arange(max_num_spans, device=embeddings.device).unsqueeze(0)

        # Create masks for valid spans
        valid_span_mask = span_indices_range < spans_per_batch.unsqueeze(1)

        # Use the mask to fill pooled_embeddings and mask
        pooled_embeddings[valid_span_mask] = span_means

        return pooled_embeddings, valid_span_mask

    def _concept_token_level_loss(self, image_token_features, text_token_features, text_token_mask, logit_scale, logit_bias=None, 
                                  span_nums=None, repeated_vector=None, negative_only=False):
        """
        Compute the token-level cosine similarity and apply the SigLIP loss.
        Args:
            image_token_features (Tensor): Image token features of shape [batch_size, img_seq_length, embedding_size].
            text_token_features (Tensor): Text token features of shape [batch_size, text_seq_length, embedding_size].
            text_token_mask (Tensor): Text token mask of shape [batch_size, text_seq_length].
        Returns:
            loss (Tensor): Computed loss value.
        """
        with torch.profiler.record_function('before_cosine_similarity'):
            batch_size, img_seq_length, embedding_size = image_token_features.shape
            _, text_seq_length, _ = text_token_features.shape

            # Normalize the features to get cosine similarity (We have already normalized the features in the forward function. So, we do not need to normalize them here)
            # merge the first and the second dimensions of the image_token_features
            image_token_features = image_token_features.view(-1, embedding_size)
            text_token_features = text_token_features[text_token_mask].view(-1, embedding_size)

        # Compute cosine similarity between image and text tokens
        with torch.profiler.record_function('cosine_similarity'):
            cosine_sim = logit_scale * torch.einsum('ie,je->ij', image_token_features, text_token_features)  # [img_seq_length, text_seq_length]
            if logit_bias is not None:
                cosine_sim += logit_bias
            cosine_sim = cosine_sim.view(batch_size, img_seq_length, -1)  # [batch_size, img_seq_length, total_text_seq_length]
        
        with torch.profiler.record_function('max_cosine_similarity'):
            max_cosine_sim = cosine_sim.max(dim=1)[0]  # [batch_size, total_text_seq_length]
        with torch.profiler.record_function('label_and_target_eye'):
            label_matrix = torch.eye(repeated_vector.size(0), len(span_nums), device=span_nums.device)[repeated_vector]  # [total_text_seq_length, batch_size]
        with torch.profiler.record_function('label_and_target_full_like'):
            target_matrix = torch.full_like(label_matrix, fill_value=-1)  # [total_text_seq_length, batch_size]
            if not negative_only:
                target_matrix += 2*label_matrix  # [total_text_seq_length, batch_size]
            target_matrix = target_matrix.t()  # [batch_size, total_text_seq_length]
        # Compute the Sigmoid Loss
        with torch.profiler.record_function('final_loss'):
            loss = -F.logsigmoid(target_matrix * max_cosine_sim).sum() / text_token_mask.sum()

        return loss

    def gather_concept_lengths(self, concept_length):
        if self.world_size > 1:
            all_concept_lengths = [torch.zeros(1).long().cuda() for _ in range(self.world_size)]
            dist.all_gather(all_concept_lengths, concept_length)
        else:
            all_concept_lengths = [concept_length]
        all_concept_lengths = torch.cat(all_concept_lengths, dim=0)
        return all_concept_lengths

    def concept_padding(self, pooled_embeddings, pooled_embedding_masks):
        if self.world_size > 1:
            all_concept_lengths = self.gather_concept_lengths(torch.tensor([pooled_embeddings.shape[1]], device=pooled_embeddings.device))
            max_concept_length = all_concept_lengths.max().item()
            if max_concept_length > pooled_embeddings.shape[1]:
                # pad the concept embeddings
                padded_embeddings = F.pad(pooled_embeddings, (0, 0, 0, max_concept_length - pooled_embeddings.shape[1], 0, 0), value=0)
                padded_masks = F.pad(pooled_embedding_masks, (0, max_concept_length - pooled_embedding_masks.shape[1], 0, 0),  value=0)
            else:
                padded_embeddings = pooled_embeddings
                padded_masks = pooled_embedding_masks
        else:
            padded_embeddings = pooled_embeddings
            padded_masks = pooled_embedding_masks
        return padded_embeddings, padded_masks
        

    def forward(self, image_features, text_features, logit_scale, logit_bias, 
                image_token_features=None, text_token_features=None, concept_logit_scale=None, concept_logit_bias=None, text_token_masks=None, meta_infos=None,
                span_nums=None, repeated_vector=None,
                output_dict=False, concept_loss_weight=1.0):
        with torch.profiler.record_function('-loss'):
            loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        # We could add a loss for the token features here
        loss_cross_gpu = True
        assert meta_infos is not None
        span_positions = meta_infos
        with torch.profiler.record_function('extract-and-pool-embeddings'):
            pooled_embeddings, pooled_embedding_masks = self.extract_and_pool_embeddings(text_token_features, span_positions, span_nums, repeated_vector)
        image_token_features = F.normalize(image_token_features, dim=-1)
        pooled_embeddings = F.normalize(pooled_embeddings, dim=-1)
        concept_loss = self._concept_token_level_loss(image_token_features, pooled_embeddings, pooled_embedding_masks, concept_logit_scale, concept_logit_bias, 
                                                      span_nums, repeated_vector, negative_only=False)

        if self.world_size > 1:
            if loss_cross_gpu:
                pooled_embeddings, pooled_embedding_masks = self.concept_padding(pooled_embeddings, pooled_embedding_masks)
                max_total_concept_num = pooled_embeddings.shape[0] * pooled_embeddings.shape[1]
                # pad repeated_vector
                repeated_vector = F.pad(repeated_vector, (0, max_total_concept_num - repeated_vector.shape[0]), value=-1)
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                if loss_cross_gpu:
                    text_token_features_to_right = text_token_features_to_left = pooled_embeddings
                    text_token_masks_to_right = text_token_masks_to_left = pooled_embedding_masks
                    span_nums_to_right = span_nums_to_left = span_nums
                    repeated_vector_to_right = repeated_vector_to_left = repeated_vector
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                        tag=TEXT_CLS_TAG,
                    )
                    if loss_cross_gpu:
                        text_token_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_token_features_to_left,
                            text_token_features_to_right,
                            tag=TEXT_TOKEN_TAG,
                        )
                        text_token_masks_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_token_masks_to_left,
                            text_token_masks_to_right,
                            tag=TEXT_MASK_TAG,
                        )
                        span_nums_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            span_nums_to_left,
                            span_nums_to_right,
                            tag=TEXT_SPAN_NUMS_TAG,
                        )
                        repeated_vector_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            repeated_vector_to_left,
                            repeated_vector_to_right,
                            tag=TEXT_REPEATED_VECTOR_TAG,
                        )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    if loss_cross_gpu:
                        assert len(text_token_features_recv) == len(text_token_masks_recv)
                        for text_token_feature, text_token_mask, _span_nums, _repeated_vector in zip(text_token_features_recv, text_token_masks_recv, span_nums_recv, repeated_vector_recv):
                            concept_loss += self._concept_token_level_loss(image_token_features,
                                                                           text_token_feature,
                                                                           text_token_mask,
                                                                           concept_logit_scale,
                                                                           concept_logit_bias,
                                                                           _span_nums,
                                                                           _repeated_vector[: _span_nums.sum()],
                                                                           negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv
                    if loss_cross_gpu:
                        text_token_features_to_left, text_token_features_to_right = text_token_features_recv
                        text_token_masks_to_left, text_token_masks_to_right = text_token_masks_recv
                        span_nums_to_left, span_nums_to_right = span_nums_recv
                        repeated_vector_to_left, repeated_vector_to_right = repeated_vector_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right, tag=TEXT_CLS_TAG)
                    if loss_cross_gpu:
                        text_token_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_features_to_right, tag=TEXT_TOKEN_TAG)
                        text_token_masks_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_masks_to_right, tag=TEXT_MASK_TAG)
                        span_nums_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank,
                            span_nums_to_right,
                            tag=TEXT_SPAN_NUMS_TAG,
                        )
                        repeated_vector_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank,
                            repeated_vector_to_right,
                            tag=TEXT_REPEATED_VECTOR_TAG,
                        )

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    if loss_cross_gpu:
                        concept_loss += self._concept_token_level_loss(image_token_features,
                                                                       text_token_features_recv,
                                                                       text_token_masks_recv,
                                                                       concept_logit_scale,
                                                                       concept_logit_bias,
                                                                       span_nums_recv,
                                                                       repeated_vector_recv[: span_nums_recv.sum()],
                                                                       negative_only=True)
            else:
                assert Exception("Cannot reach here")
                text_features_to_right = text_features
                if loss_cross_gpu:
                    text_token_features_to_right = pooled_embeddings
                    text_token_masks_to_right = pooled_embedding_masks
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right, tag=TEXT_CLS_TAG)
                    if loss_cross_gpu:
                        text_token_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_features_to_right, tag=TEXT_TOKEN_TAG)
                        text_token_masks_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_masks_to_right, tag=TEXT_MASK_TAG)
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    if loss_cross_gpu:
                        concept_loss += self._concept_token_level_loss(image_token_features,
                                                                       text_token_features_from_left,
                                                                       text_token_masks_from_left,
                                                                       concept_logit_scale,
                                                                       concept_logit_bias,
                                                                       negative_only=True)
                    text_features_to_right = text_features_from_left
                    if loss_cross_gpu:
                        text_token_features_to_right = text_token_features_from_left
                        text_token_masks_to_right = text_token_masks_from_left
        return {"contrastive_loss": loss, "concept_loss": concept_loss_weight * concept_loss} if output_dict else (loss, concept_loss_weight * concept_loss)


class SigLipConceptLSELoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def extract_and_pool_embeddings(self, embeddings, span_positions, span_nums, repeated_vector):
        """
        Extracts embeddings for specified spans and performs mean pooling.

        Parameters:
        - embeddings (torch.Tensor): Tensor of shape [batch_size, sequence_length, embedding_size].
        - span_positions (torch.Tensor): Tensor of shape [batch_size, concept_max_num, 2], where 2 is the start and end positions of the span.

        Returns:
        - pooled_embeddings (torch.Tensor): Tensor of shape [batch_size, num_spans, embedding_size].
        """
        span_positions = span_positions + 1  # We add 1 to the start and end positions to account for the [CLS] token
        batch_size, sequence_length, embedding_size = embeddings.shape
        padded_span_mask = span_positions.sum(dim=-1).bool()
        spans_per_batch = span_nums
        # # Get batch indices and span start/end indices
        batch_indices = repeated_vector

        padded_span_mask_x, padded_span_mask_y = torch.nonzero(padded_span_mask, as_tuple=True)
        start_indices, end_indices = span_positions[padded_span_mask_x, padded_span_mask_y].unbind(dim=-1)
        # Create a tensor for the spans
        max_num_spans = span_positions.size(1)
        max_span_length = (end_indices - start_indices).max().item()

        # Initialize the pooled embeddings tensor and mask
        pooled_embeddings = torch.zeros(batch_size, max_num_spans, embedding_size, device=embeddings.device)

        # Create indices for gathering
        span_range = torch.arange(max_span_length, device=embeddings.device).unsqueeze(0)  # Shape: [1, max_span_length]

        # Prepare the indices for advanced indexing
        batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, max_span_length)
        start_indices_expanded = start_indices.unsqueeze(1).expand(-1, max_span_length)
        end_indices_expanded = end_indices.unsqueeze(1).expand(-1, max_span_length)

        # Calculate the actual indices
        actual_indices = start_indices_expanded + span_range
        # Create a mask for valid indices
        valid_mask = actual_indices < end_indices_expanded

        # Clamp the actual indices to avoid out-of-bounds
        actual_indices = torch.clamp(actual_indices, min=0, max=sequence_length - 1)  # Ensure no out-of-bounds

        # Gather embeddings for each span
        gathered_embeddings = embeddings[batch_indices_expanded, actual_indices]

        # Zero out invalid indices
        gathered_embeddings[~valid_mask] = 0  # Set invalid indices to zero

        # Perform mean pooling, ignoring zeros
        span_means = gathered_embeddings.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).float()

        # Create indices for batch and span positions
        span_indices_range = torch.arange(max_num_spans, device=embeddings.device).unsqueeze(0)

        # Create masks for valid spans
        valid_span_mask = span_indices_range < spans_per_batch.unsqueeze(1)

        # Use the mask to fill pooled_embeddings and mask
        pooled_embeddings[valid_span_mask] = span_means

        return pooled_embeddings, valid_span_mask

    def _concept_token_level_loss(self, image_token_features, text_token_features, text_token_mask, logit_scale, logit_bias=None, 
                                  span_nums=None, repeated_vector=None, negative_only=False):
        """
        Compute the token-level cosine similarity and apply the SigLIP loss.
        Args:
            image_token_features (Tensor): Image token features of shape [batch_size, img_seq_length, embedding_size].
            text_token_features (Tensor): Text token features of shape [batch_size, text_seq_length, embedding_size].
            text_token_mask (Tensor): Text token mask of shape [batch_size, text_seq_length].
        Returns:
            loss (Tensor): Computed loss value.
        """
        batch_size, img_seq_length, embedding_size = image_token_features.shape
        _, text_seq_length, _ = text_token_features.shape

        # Normalize the features to get cosine similarity (We have already normalized the features in the forward function. So, we do not need to normalize them here)
        # merge the first and the second dimensions of the image_token_features
        image_token_features = image_token_features.view(-1, embedding_size)
        text_token_features = text_token_features[text_token_mask].view(-1, embedding_size)

        # Compute cosine similarity between image and text tokens
        cosine_sim = logit_scale * torch.einsum('ie,je->ij', image_token_features, text_token_features)  # [img_seq_length, text_seq_length]
        if logit_bias is not None:
            cosine_sim += logit_bias
        cosine_sim = cosine_sim.view(batch_size, img_seq_length, -1)  # [batch_size, img_seq_length, total_text_seq_length]

        max_cosine_sim = cosine_sim.max(dim=1)[0]  # [batch_size, total_text_seq_length]
        '''
        label_matrix = torch.eye(repeated_vector.size(0), len(span_nums), device=span_nums.device)[repeated_vector]  # [total_text_seq_length, batch_size]
        target_matrix = torch.full_like(label_matrix, fill_value=-1)  # [total_text_seq_length, batch_size]
        if not negative_only:
            target_matrix += 2*label_matrix  # [total_text_seq_length, batch_size]
        target_matrix = target_matrix.t()  # [batch_size, total_text_seq_length]
        # Compute the Sigmoid Loss
        with torch.profiler.record_function('final_loss'):
            loss = -F.logsigmoid(target_matrix * max_cosine_sim).sum() / text_token_mask.sum()
        '''
        sim_matrix = torch.full((batch_size, batch_size, text_seq_length), float('-inf'), device=max_cosine_sim.device, dtype=max_cosine_sim.dtype)
        # extract the positions of valid similarities
        sim_matrix[:, text_token_mask] = max_cosine_sim
        # normalize the similarity between each image-text pair by the number of concepts
        sim_matrix = torch.log(torch.exp(sim_matrix).sum(dim=-1) / span_nums)
        labels = self.get_ground_truth(
            image_token_features.device,
            image_token_features.dtype,
            batch_size,
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * sim_matrix).sum() / batch_size
        return loss

    def gather_concept_lengths(self, concept_length):
        if self.world_size > 1:
            all_concept_lengths = [torch.zeros(1).long().cuda() for _ in range(self.world_size)]
            dist.all_gather(all_concept_lengths, concept_length)
        else:
            all_concept_lengths = [concept_length]
        all_concept_lengths = torch.cat(all_concept_lengths, dim=0)
        return all_concept_lengths

    def concept_padding(self, pooled_embeddings, pooled_embedding_masks):
        if self.world_size > 1:
            all_concept_lengths = self.gather_concept_lengths(torch.tensor([pooled_embeddings.shape[1]], device=pooled_embeddings.device))
            max_concept_length = all_concept_lengths.max().item()
            if max_concept_length > pooled_embeddings.shape[1]:
                # pad the concept embeddings
                padded_embeddings = F.pad(pooled_embeddings, (0, 0, 0, max_concept_length - pooled_embeddings.shape[1], 0, 0), value=0)
                padded_masks = F.pad(pooled_embedding_masks, (0, max_concept_length - pooled_embedding_masks.shape[1], 0, 0),  value=0)
            else:
                padded_embeddings = pooled_embeddings
                padded_masks = pooled_embedding_masks
        else:
            padded_embeddings = pooled_embeddings
            padded_masks = pooled_embedding_masks
        return padded_embeddings, padded_masks
        

    def forward(self, image_features, text_features, logit_scale, logit_bias, 
                image_token_features=None, text_token_features=None, concept_logit_scale=None, concept_logit_bias=None, text_token_masks=None, meta_infos=None,
                span_nums=None, repeated_vector=None,
                output_dict=False, concept_loss_weight=1.0):
        with torch.profiler.record_function('-loss'):
            loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        # We could add a loss for the token features here
        loss_cross_gpu = True
        assert meta_infos is not None
        span_positions = meta_infos
        with torch.profiler.record_function('extract-and-pool-embeddings'):
            pooled_embeddings, pooled_embedding_masks = self.extract_and_pool_embeddings(text_token_features, span_positions, span_nums, repeated_vector)
        image_token_features = F.normalize(image_token_features, dim=-1)
        pooled_embeddings = F.normalize(pooled_embeddings, dim=-1)
        concept_loss = self._concept_token_level_loss(image_token_features, pooled_embeddings, pooled_embedding_masks, concept_logit_scale, concept_logit_bias, 
                                                      span_nums, repeated_vector, negative_only=False)

        if self.world_size > 1:
            if loss_cross_gpu:
                pooled_embeddings, pooled_embedding_masks = self.concept_padding(pooled_embeddings, pooled_embedding_masks)
                max_total_concept_num = pooled_embeddings.shape[0] * pooled_embeddings.shape[1]
                # pad repeated_vector
                repeated_vector = F.pad(repeated_vector, (0, max_total_concept_num - repeated_vector.shape[0]), value=-1)
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                if loss_cross_gpu:
                    text_token_features_to_right = text_token_features_to_left = pooled_embeddings
                    text_token_masks_to_right = text_token_masks_to_left = pooled_embedding_masks
                    span_nums_to_right = span_nums_to_left = span_nums
                    repeated_vector_to_right = repeated_vector_to_left = repeated_vector
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                        tag=TEXT_CLS_TAG,
                    )
                    if loss_cross_gpu:
                        text_token_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_token_features_to_left,
                            text_token_features_to_right,
                            tag=TEXT_TOKEN_TAG,
                        )
                        text_token_masks_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_token_masks_to_left,
                            text_token_masks_to_right,
                            tag=TEXT_MASK_TAG,
                        )
                        span_nums_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            span_nums_to_left,
                            span_nums_to_right,
                            tag=TEXT_SPAN_NUMS_TAG,
                        )
                        repeated_vector_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            repeated_vector_to_left,
                            repeated_vector_to_right,
                            tag=TEXT_REPEATED_VECTOR_TAG,
                        )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    if loss_cross_gpu:
                        assert len(text_token_features_recv) == len(text_token_masks_recv)
                        for text_token_feature, text_token_mask, _span_nums, _repeated_vector in zip(text_token_features_recv, text_token_masks_recv, span_nums_recv, repeated_vector_recv):
                            concept_loss += self._concept_token_level_loss(image_token_features,
                                                                           text_token_feature,
                                                                           text_token_mask,
                                                                           concept_logit_scale,
                                                                           concept_logit_bias,
                                                                           _span_nums,
                                                                           _repeated_vector[: _span_nums.sum()],
                                                                           negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv
                    if loss_cross_gpu:
                        text_token_features_to_left, text_token_features_to_right = text_token_features_recv
                        text_token_masks_to_left, text_token_masks_to_right = text_token_masks_recv
                        span_nums_to_left, span_nums_to_right = span_nums_recv
                        repeated_vector_to_left, repeated_vector_to_right = repeated_vector_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right, tag=TEXT_CLS_TAG)
                    if loss_cross_gpu:
                        text_token_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_features_to_right, tag=TEXT_TOKEN_TAG)
                        text_token_masks_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_masks_to_right, tag=TEXT_MASK_TAG)
                        span_nums_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank,
                            span_nums_to_right,
                            tag=TEXT_SPAN_NUMS_TAG,
                        )
                        repeated_vector_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank,
                            repeated_vector_to_right,
                            tag=TEXT_REPEATED_VECTOR_TAG,
                        )

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    if loss_cross_gpu:
                        # print('text_token_features_recv.shape:', text_token_features_recv.shape)
                        concept_loss += self._concept_token_level_loss(image_token_features,
                                                                       text_token_features_recv,
                                                                       text_token_masks_recv,
                                                                       concept_logit_scale,
                                                                       concept_logit_bias,
                                                                       span_nums_recv,
                                                                       repeated_vector_recv[: span_nums_recv.sum()],
                                                                       negative_only=True)
            else:
                assert Exception("Cannot reach here")
                text_features_to_right = text_features
                if loss_cross_gpu:
                    text_token_features_to_right = pooled_embeddings
                    text_token_masks_to_right = pooled_embedding_masks
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right, tag=TEXT_CLS_TAG)
                    if loss_cross_gpu:
                        text_token_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_features_to_right, tag=TEXT_TOKEN_TAG)
                        text_token_masks_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_masks_to_right, tag=TEXT_MASK_TAG)
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    if loss_cross_gpu:
                        concept_loss += self._concept_token_level_loss(image_token_features,
                                                                       text_token_features_from_left,
                                                                       text_token_masks_from_left,
                                                                       concept_logit_scale,
                                                                       concept_logit_bias,
                                                                       negative_only=True)
                    text_features_to_right = text_features_from_left
                    if loss_cross_gpu:
                        text_token_features_to_right = text_token_features_from_left
                        text_token_masks_to_right = text_token_masks_from_left
        return {"contrastive_loss": loss, "concept_loss": concept_loss_weight * concept_loss} if output_dict else (loss, concept_loss_weight * concept_loss)


class SigLipTokenLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def extract_and_pool_embeddings(self, embeddings, span_positions):
        """
        Extracts embeddings for specified spans and performs mean pooling.

        Parameters:
        - embeddings (torch.Tensor): Tensor of shape [batch_size, sequence_length, embedding_size].
        - span_positions (list): List of spans for each batch, where each span is [start_position, end_position].

        Returns:
        - pooled_embeddings (torch.Tensor): Tensor of shape [batch_size, num_spans, embedding_size].
        """
        batch_size, sequence_length, embedding_size = embeddings.shape

        # Prepare lists to collect spans
        all_spans = []
        spans_per_batch = torch.zeros(batch_size, dtype=torch.int, device=embeddings.device)
        for batch_index, spans in enumerate(span_positions):
            for start, end in spans:
                # TODO: Pay attention! We will remove the None check once we have extracted all the correct span positions in the future PR (but end + 1 >= sequence_length is still needed)
                if start is None or end is None or end + 1 >= sequence_length:
                    continue
                assert start < end, f"Start position should be less than end position. {start} >= {end}"
                all_spans.append((batch_index, start + 1, end + 1)) # We add 1 to the start and end positions to account for the [CLS] token
                spans_per_batch[batch_index] += 1  # Count spans for this batch
        all_spans_tensor = torch.tensor(all_spans, dtype=torch.long, device=embeddings.device)

        # Get batch indices and span start/end indices
        batch_indices = all_spans_tensor[:, 0]
        start_indices = all_spans_tensor[:, 1]
        end_indices = all_spans_tensor[:, 2]

        # Create a tensor for the spans
        num_spans = all_spans_tensor.size(0)
        max_num_spans = spans_per_batch.max().item()
        max_span_length = (end_indices - start_indices).max().item()

        # Initialize the pooled embeddings tensor and mask
        pooled_embeddings = torch.zeros(batch_size, max_num_spans, embedding_size, device=embeddings.device)

        # Create indices for gathering
        span_length = end_indices - start_indices
        span_range = torch.arange(max_span_length, device=embeddings.device).unsqueeze(0)  # Shape: [1, max_span_length]

        # Prepare the indices for advanced indexing
        batch_indices_expanded = batch_indices.unsqueeze(1).expand(-1, max_span_length)
        start_indices_expanded = start_indices.unsqueeze(1).expand(-1, max_span_length)
        end_indices_expanded = end_indices.unsqueeze(1).expand(-1, max_span_length)

        # Calculate the actual indices
        actual_indices = start_indices_expanded + span_range
        # Create a mask for valid indices
        valid_mask = actual_indices < end_indices_expanded

        # Clamp the actual indices to avoid out-of-bounds
        actual_indices = torch.clamp(actual_indices, min=0, max=sequence_length - 1)  # Ensure no out-of-bounds

        # Gather embeddings for each span
        gathered_embeddings = embeddings[batch_indices_expanded, actual_indices]

        # Zero out invalid indices
        gathered_embeddings[~valid_mask] = 0  # Set invalid indices to zero

        # Perform mean pooling, ignoring zeros
        span_means = gathered_embeddings.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).float()

        # Create indices for batch and span positions
        # batch_indices_range = torch.arange(batch_size, device=embeddings.device).unsqueeze(1)
        span_indices_range = torch.arange(max_num_spans, device=embeddings.device).unsqueeze(0)

        # Create masks for valid spans
        valid_span_mask = span_indices_range < spans_per_batch.unsqueeze(1)

        # Use the mask to fill pooled_embeddings and mask
        pooled_embeddings[valid_span_mask] = span_means

        return pooled_embeddings, valid_span_mask

    def _concept_token_level_loss(self, image_token_features, text_token_features, text_token_mask, logit_scale, logit_bias=None, negative_only=False):
        """
        Compute the token-level cosine similarity and apply the SigLIP loss.
        Args:
            image_token_features (Tensor): Image token features of shape [batch_size, img_seq_length, embedding_size].
            text_token_features (Tensor): Text token features of shape [batch_size, text_seq_length, embedding_size].
            text_token_mask (Tensor): Text token mask of shape [batch_size, text_seq_length].
        Returns:
            loss (Tensor): Computed loss value.
        """
        batch_size, img_seq_length, embedding_size = image_token_features.shape
        _, text_seq_length, _ = text_token_features.shape

        # Normalize the features to get cosine similarity
        image_token_features = F.normalize(image_token_features, dim=-1)
        text_token_features = F.normalize(text_token_features, dim=-1)

        # merge the first and the second dimensions of the image_token_features
        image_token_features = image_token_features.view(-1, embedding_size)
        text_token_features = text_token_features[text_token_mask].view(-1, embedding_size)

        # Compute cosine similarity between image and text tokens
        cosine_sim = logit_scale * torch.einsum('ie,je->ij', image_token_features, text_token_features)  # [img_seq_length, text_seq_length]
        if logit_bias is not None:
            cosine_sim += logit_bias
        cosine_sim = cosine_sim.view(batch_size, img_seq_length, -1)  # [batch_size, img_seq_length, total_text_seq_length]
        
        max_cosine_sim = cosine_sim.max(dim=1)[0]  # [batch_size, total_text_seq_length]

        span_nums = text_token_mask.sum(dim=1)  # [batch_size]
        repeated_vector = torch.repeat_interleave(torch.arange(len(span_nums), device=span_nums.device), span_nums)  # [total_text_seq_length]
        label_matrix = torch.eye(repeated_vector.size(0), len(span_nums), device=span_nums.device)[repeated_vector]  # [total_text_seq_length, batch_size]
        target_matrix = torch.full_like(label_matrix, fill_value=-1)  # [total_text_seq_length, batch_size]
        if not negative_only:
            target_matrix += 2*label_matrix  # [total_text_seq_length, batch_size]
        target_matrix = target_matrix.t()  # [batch_size, total_text_seq_length]
        # Compute the Sigmoid Loss
        # TODO: We have not yet determine the denominator of the loss, which should be the batch_size or the total number of text span tokens or the multiplication of the two
        loss = -F.logsigmoid(target_matrix * max_cosine_sim).sum() / text_token_mask.sum()

        return loss

    def gather_concept_lengths(self, concept_length):
        if self.world_size > 1:
            all_concept_lengths = [torch.zeros(1).long().cuda() for _ in range(self.world_size)]
            dist.all_gather(all_concept_lengths, concept_length)
        else:
            all_concept_lengths = [concept_length]
        all_concept_lengths = torch.cat(all_concept_lengths, dim=0)
        return all_concept_lengths

    def concept_padding(self, pooled_embeddings, pooled_embedding_masks):
        if self.world_size > 1:
            all_concept_lengths = self.gather_concept_lengths(torch.tensor([pooled_embeddings.shape[1]], device=pooled_embeddings.device))
            max_concept_length = all_concept_lengths.max().item()
            if max_concept_length > pooled_embeddings.shape[1]:
                # pad the concept embeddings
                padded_embeddings = F.pad(pooled_embeddings, (0, 0, 0, max_concept_length - pooled_embeddings.shape[1], 0, 0), value=0)
                padded_masks = F.pad(pooled_embedding_masks, (0, max_concept_length - pooled_embedding_masks.shape[1], 0, 0),  value=0)
            else:
                padded_embeddings = pooled_embeddings
                padded_masks = pooled_embedding_masks
        else:
            padded_embeddings = pooled_embeddings
            padded_masks = pooled_embedding_masks
        return padded_embeddings, padded_masks
        

    def forward(self, image_features, text_features, logit_scale, logit_bias, 
                image_token_features, text_token_features, text_token_masks, concept_logit_scale, concept_logit_bias, meta_infos=None,
                output_dict=False, concept_loss_weight=1.0):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)
        # We could add a loss for the token features here
        pooled_embeddings, pooled_embedding_masks = text_token_features, text_token_masks
        concept_loss = self._concept_token_level_loss(image_token_features, pooled_embeddings, pooled_embedding_masks, concept_logit_scale, concept_logit_bias, negative_only=False)

        if self.world_size > 1:
            pooled_embeddings, pooled_embedding_masks = self.concept_padding(pooled_embeddings, pooled_embedding_masks)
            # exchange text features w/ neighbour world_size - 1 times
            loss_cross_gpu = True # FIXME: implement this
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                text_token_features_to_right = text_token_features_to_left = pooled_embeddings
                text_token_masks_to_right = text_token_masks_to_left = pooled_embedding_masks
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                        tag=TEXT_CLS_TAG,
                    )
                    if loss_cross_gpu:
                        text_token_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_token_features_to_left,
                            text_token_features_to_right,
                            tag=TEXT_TOKEN_TAG,
                        )
                        text_token_masks_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_token_masks_to_left,
                            text_token_masks_to_right,
                            tag=TEXT_MASK_TAG,
                        )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    if loss_cross_gpu:
                        assert len(text_token_features_recv) == len(text_token_masks_recv)
                        for text_token_feature, text_token_mask in zip(text_token_features_recv, text_token_masks_recv):
                            concept_loss += self._concept_token_level_loss(image_token_features,
                                                                           text_token_feature,
                                                                           text_token_mask,
                                                                           concept_logit_scale,
                                                                           concept_logit_bias,
                                                                           negative_only=True)
                    text_features_to_left, text_features_to_right = text_features_recv
                    if loss_cross_gpu:
                        pooled_embeddings_to_left, pooled_embeddings_to_right = text_token_features_recv
                        pooled_embedding_masks_to_left, pooled_embedding_masks_to_right = text_token_masks_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right, tag=TEXT_CLS_TAG)
                    if loss_cross_gpu:
                        text_token_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_features_to_right, tag=TEXT_TOKEN_TAG)
                        text_token_masks_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_masks_to_right, tag=TEXT_MASK_TAG)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    if loss_cross_gpu:
                        concept_loss += self._concept_token_level_loss(image_token_features,
                                                                       text_token_features_recv,
                                                                       text_token_masks_recv,
                                                                       concept_logit_scale,
                                                                       concept_logit_bias,
                                                                       negative_only=True)
            else:
                text_features_to_right = text_features
                if loss_cross_gpu:
                    text_token_features_to_right = pooled_embeddings
                    text_token_masks_to_right = pooled_embedding_masks
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right, tag=TEXT_CLS_TAG)
                    if loss_cross_gpu:
                        text_token_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_features_to_right, tag=TEXT_TOKEN_TAG)
                        text_token_masks_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, text_token_masks_to_right, tag=TEXT_MASK_TAG)
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    if loss_cross_gpu:
                        concept_loss += self._concept_token_level_loss(image_token_features,
                                                                       text_token_features_from_left,
                                                                       text_token_masks_from_left,
                                                                       concept_logit_scale,
                                                                       concept_logit_bias,
                                                                       negative_only=True)
                    text_features_to_right = text_features_from_left
                    if loss_cross_gpu:
                        text_token_features_to_right = text_token_features_from_left
                        text_token_masks_to_right = text_token_masks_from_left
        return {"contrastive_loss": loss, "concept_loss": concept_loss_weight * concept_loss} if output_dict else (loss, 1.0 * concept_loss)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False,):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False, **kwargs):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
