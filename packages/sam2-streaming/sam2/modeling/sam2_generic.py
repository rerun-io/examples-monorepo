import torch
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

from sam2.modeling.sam2_utils import get_1d_sine_pe
from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.memory import ObjectMemory


class SAM2Generic(SAM2Base):

    def __init__(
        self,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        non_overlap_masks=False,
        **kwargs,
    ) -> None:
        """
        SAM2Generic is a class that extends SAM2Base to provide easier APIs for generic segmentation tasks.

        Arguments:
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__(**kwargs)
        self._transforms = SAM2Transforms(
            resolution=self.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.mask_threshold = mask_threshold
        self.non_overlap_masks = non_overlap_masks

        self.empty_prompt_embeddings = self.encode_prompts()

    def _prepare_images(
        self, img: torch.Tensor | list[torch.Tensor], scale: bool = True
    ):

        # If we have a list of images (potentially of different sizes), we apply the transforms to each image
        # and then concatenate them along the batch dimension.
        img_list = [img] if not isinstance(img, (list, tuple)) else img

        for i, img in enumerate(img_list):
            assert img.ndim in [
                3,
                4,
            ], f"Expected image to be of shape (B, C, H, W) or (C, H, W), got {img.shape}"
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.dtype == torch.uint8 and scale:
                img = img.float() / 255.0
            img_list[i] = self._transforms.transforms(img)

        return torch.cat(img_list, dim=0)

    @torch.inference_mode()
    def encode_image(
        self, image: torch.Tensor | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the image for the SAM-2 model.

        Args:
            image (torch.Tensor | list[torch.Tensor]): The image or list of images to encode.

        Returns:
            img_embeddings (torch.Tensor): The image embeddings (the last one being the lowest resolution).
            img_pos_embeddings (torch.Tensor): The image position embeddings (the last one being the lowest resolution).
        """
        img_batch = self._prepare_images(image)

        backbone_out = self.image_encoder(img_batch)

        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )

        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        img_embeddings = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        img_pos_embeddings = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        return img_embeddings, img_pos_embeddings

    @torch.inference_mode()
    def encode_memory(
        self,
        img_embeddings: list[torch.Tensor],
        masks_logits: torch.Tensor,
        obj_score_logits: torch.Tensor,
        is_prompt: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the image and its prediction into a memory.

        Args:
            img_embeddings (list[torch.Tensor]): The image embeddings.
            masks_high_res_logits (torch.Tensor): The high-resolution mask logits.
            object_score_logits (torch.Tensor): The object score logits.
            is_prompt (torch.BoolTensor): Whether the masks are from a user prompt or from a SAM prediction.

        Returns:
            memory_embeddings (torch.Tensor): The encoded memory embeddings.
            memory_pos_embeddings (torch.Tensor): The encoded memory position embeddings.
        """

        assert [
            t.ndim == 4 for t in img_embeddings
        ], f"Expected all levels of img_embeddings to be of shape (B, C, H, W), got {[t.shape for t in img_embeddings]}"
        B = img_embeddings[0].shape[0]
        assert (
            masks_logits.ndim == 4 and masks_logits.shape[0] == B
        ), f"Expected masks_logits to be of shape (B, C, H, W), got {masks_logits.shape}"
        assert obj_score_logits.shape == (
            B,
            1,
        ), f"Expected obj_score_logits to be of shape ({B}, 1), got {obj_score_logits.shape}"
        assert is_prompt.shape == (
            B,
        ), f"Expected is_prompt to be of shape ({B},), got {is_prompt.shape}"

        low_res_img_embeddings = img_embeddings[-1]

        if self.non_overlap_masks_for_mem_enc and not self.training:
            masks_logits = self._apply_non_overlapping_constraints(masks_logits)

        masks_logits = self._transforms.downscale_masks_logits(masks_logits)

        # Scale the raw mask logits with a temperature before applying sigmoid
        binarize = (
            self.binarize_mask_from_pts_for_mem_enc & is_prompt & (not self.training)
        )

        mask_for_mem = torch.where(
            binarize.reshape((-1, 1, 1, 1)),
            (masks_logits > self.mask_threshold).float(),
            torch.sigmoid(
                masks_logits
            ),  # Apply sigmoid on the raw mask logits to turn them into range (0, 1)
        )

        # Apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            low_res_img_embeddings,
            mask_for_mem,
            skip_mask_sigmoid=True,  # sigmoid already applied
        )

        memory_embeddings = maskmem_out["vision_features"]
        memory_pos_embeddings = maskmem_out["vision_pos_enc"][0]

        # Add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (obj_score_logits > 0).float()
            memory_embeddings += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *memory_embeddings.shape
            )

        return memory_embeddings, memory_pos_embeddings

    @torch.inference_mode()
    def _prepare_obj_ptrs_for_memory_conditioning(
        self,
        current_frame_idx: int,
        obj_ptrs: torch.Tensor,
        obj_ptrs_frame_idx: list[int],
        reverse_time: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the object pointers into a memory.
        """
        assert (
            obj_ptrs.ndim == 3
        ), f"Expected obj_ptrs to be of shape (ptr_seq_len, B, C), got {obj_ptrs.shape}"

        B = obj_ptrs.shape[1]

        tpos_sign_mul = -1 if reverse_time else 1

        obj_ptrs_frame_idx = torch.tensor(obj_ptrs_frame_idx, device=obj_ptrs.device)

        obj_tpos_rel = (
            (current_frame_idx - obj_ptrs_frame_idx) * tpos_sign_mul
            if self.use_signed_tpos_enc_to_obj_ptrs
            else torch.abs(current_frame_idx - obj_ptrs_frame_idx)
        )

        if self.add_tpos_enc_to_obj_ptrs:
            t_diff_max = self.max_obj_ptrs_in_encoder - 1
            tpos_dim = (
                self.hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
            )

            obj_tpos_rel = get_1d_sine_pe(obj_tpos_rel / t_diff_max, dim=tpos_dim)
            obj_tpos_rel = self.obj_ptr_tpos_proj.forward(obj_tpos_rel)
            obj_tpos_rel = obj_tpos_rel.unsqueeze(1).expand(-1, B, self.mem_dim)
        else:
            obj_tpos_rel = obj_ptrs.new_zeros(len(obj_tpos_rel), B, self.mem_dim)

        if self.mem_dim < self.hidden_dim:
            # Split a pointer into (self.hidden_dim // self.mem_dim) tokens for self.mem_dim < self.hidden_dim
            obj_ptrs = obj_ptrs.reshape(
                -1,
                B,
                self.hidden_dim // self.mem_dim,
                self.mem_dim,
            )
            obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
            obj_tpos_rel = obj_tpos_rel.repeat_interleave(
                self.hidden_dim // self.mem_dim, dim=0
            )

        return obj_ptrs, obj_tpos_rel

    @torch.inference_mode()
    def _prepare_memory_for_memory_conditioning(
        self,
        t_pos: int,
        memory_embeddings: torch.Tensor,
        memory_pos_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory_embeddings = memory_embeddings.flatten(2).permute(
            2, 0, 1
        )  # (B, C, H, W) -> (H*W, B, C)
        memory_pos_embeddings = memory_pos_embeddings.flatten(2).permute(
            2, 0, 1
        )  # (B, C, H, W) -> (H*W, B, C)

        # Add temporal positional encoding
        memory_tpos_embeddings = (
            memory_pos_embeddings + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
        )
        return memory_embeddings, memory_tpos_embeddings

    @torch.inference_mode()
    def condition_image_embeddings_on_memories(
        self,
        frame_idx: int,
        img_embeddings: list[torch.Tensor],
        img_pos_embeddings: list[torch.Tensor],
        conditional_memories: list[ObjectMemory],
        non_conditional_memories: list[ObjectMemory],
        ptr_memories: list[ObjectMemory],
        reverse_time: bool = False,
    ) -> list[torch.Tensor]:
        """
        Condition the image embeddings on the memory embeddings.

        Note: the non conditional memories are ordered by order of importance, i.e. the first non conditional memory is the most important one.
        For example, if you use temporal memory, the first non conditional memory is the one that is closest to the current frame, and the last one is the one that is farthest.

        Args:
            frame_idx (int): The index of the current frame.
            img_embeddings (list[torch.Tensor]): The image embeddings.
            img_pos_embeddings (list[torch.Tensor]): The image position embeddings.
            conditional_memories (list[ObjectMemory]): The conditional memories.
            non_conditional_memories (list[ObjectMemory]): The non conditional memories.
            ptr_memories (list[ObjectMemory]): The pointer memories.
            reverse_time (bool): Whether to reverse the time.

        Returns:
            list[torch.Tensor]: The conditioned image embeddings.
        """

        # Stack the pointer memories along a new sequence dimension (ptr_seq_len, B, C)
        if len(ptr_memories) > 0:
            obj_ptrs_seq = torch.stack([m.ptr for m in ptr_memories], dim=0)
        else:
            B = img_embeddings[0].shape[0]
            obj_ptrs_seq = torch.zeros(
                (
                    0,
                    B,
                    self.mem_dim,
                ),
                device=self.device,
            )
        obj_ptrs_frame_indices = [m.frame_idx for m in ptr_memories]

        assert (
            self.max_cond_frames_in_attn == -1
            or len(conditional_memories) <= self.max_cond_frames_in_attn
        ), f"Expected at most {self.max_cond_frames_in_attn} conditional memories, got {len(conditional_memories)}"
        assert (
            len(non_conditional_memories) <= self.num_maskmem - 1
        ), f"Expected at most {self.num_maskmem - 1} non-conditional memories, got {len(non_conditional_memories)}"
        assert (
            self.max_obj_ptrs_in_encoder == -1
            or len(ptr_memories) <= self.max_obj_ptrs_in_encoder
        ), f"Expected at most {self.max_obj_ptrs_in_encoder} object pointer memories, got {len(ptr_memories)}"

        low_res_img_embeddings = img_embeddings[-1]
        low_res_img_pos_embeddings = img_pos_embeddings[-1]
        high_res_img_embeddings = img_embeddings[:-1]

        B, C, H, W = low_res_img_embeddings.shape

        n_conditional_memories = len(conditional_memories)
        n_non_conditional_memories = len(non_conditional_memories)
        n_ptrs_memories = len(obj_ptrs_seq)

        if (
            n_conditional_memories == 0
            and n_non_conditional_memories == 0
            and n_ptrs_memories == 0
        ):
            # We don't have any memories, we add the no-mem embedding
            if self.directly_add_no_mem_embed:
                # Directly add the no-mem embedding (instead of using the transformer encoder)
                low_res_img_embeddings = low_res_img_embeddings.flatten(2).permute(
                    2, 0, 1
                )  # (B, C, H, W) -> (H*W, B, C)
                low_res_img_embeddings = low_res_img_embeddings + self.no_mem_embed
                low_res_img_embeddings = low_res_img_embeddings.permute(1, 2, 0).view(
                    B, C, H, W
                )
                return low_res_img_embeddings

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            memories = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            memories_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]
            num_obj_ptr_tokens = 0

        else:

            memories = []
            memories_pos_embed = []

            # Add conditional memories (prompt from the user)
            for cond_mem in conditional_memories:
                memory_embeddings, memory_tpos_embeddings = (
                    self._prepare_memory_for_memory_conditioning(
                        0, cond_mem.memory_embeddings, cond_mem.memory_pos_embeddings
                    )
                )
                memories.append(memory_embeddings)
                memories_pos_embed.append(memory_tpos_embeddings)

            # Add non-conditional memories (memory from previous frames, or other depending on the memory strategy)
            for i, non_cond_mem in enumerate(non_conditional_memories):
                t_pos = i + 1
                memory_embeddings, memory_tpos_embeddings = (
                    self._prepare_memory_for_memory_conditioning(
                        t_pos,
                        non_cond_mem.memory_embeddings,
                        non_cond_mem.memory_pos_embeddings,
                    )
                )
                memories.append(memory_embeddings)
                memories_pos_embed.append(memory_tpos_embeddings)
            if self.use_obj_ptrs_in_encoder:

                obj_ptrs_enc, obj_pos_enc = (
                    self._prepare_obj_ptrs_for_memory_conditioning(
                        frame_idx, obj_ptrs_seq, obj_ptrs_frame_indices, reverse_time
                    )
                )

                memories.append(obj_ptrs_enc)
                memories_pos_embed.append(obj_pos_enc)
                num_obj_ptr_tokens = obj_ptrs_enc.shape[0]

        memory = torch.cat(memories, dim=0)
        memory_pos_embed = torch.cat(memories_pos_embed, dim=0)

        low_res_img_embeddings = low_res_img_embeddings.flatten(2).permute(
            2, 0, 1
        )  # (B, C, H, W) -> (H*W, B, C)
        low_res_img_pos_embeddings = low_res_img_pos_embeddings.flatten(2).permute(
            2, 0, 1
        )  # (B, C, H, W) -> (H*W, B, C)

        low_res_img_embeddings_with_mem = self.memory_attention(
            curr=low_res_img_embeddings,
            curr_pos=low_res_img_pos_embeddings,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        low_res_img_embeddings_with_mem = low_res_img_embeddings_with_mem.permute(
            1, 2, 0
        ).view(B, C, H, W)

        return high_res_img_embeddings + [low_res_img_embeddings_with_mem]

    @torch.inference_mode()
    def encode_prompts(
        self,
        orig_hw: tuple[int, int] | None = None,
        batch_size: int = 1,
        points_coords: torch.Tensor | None = None,
        points_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        masks_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the prompts for the SAM-2 model.

        Args:
            orig_hw (tuple[int, int]): The original height and width of the image.
            batch_size (int): The batch size of the prompts.
            points_coords (torch.Tensor | None): The coordinates of the points to encode. Shape: (B, N, 2) with N being the number of points and the last dimension being (x, y).
            points_labels (torch.Tensor | None): The labels of the points to encode. Shape: (B, N).
            boxes (torch.Tensor | None): The boxes to encode. Shape: (B, 4) with the last dimension being (x1, y1, x2, y2).
            masks_logits (torch.Tensor | None): The masks logits to encode. Shape: (B, H, W).

        Returns:
            prompt_embeddings (tuple[torch.Tensor, torch.Tensor]): The sparse and dense prompt embeddings.
        """

        if points_coords is not None or boxes is not None:
            assert (
                orig_hw is not None
            ), "Expected orig_hw to be provided if points_coords or boxes are provided"

        points = None

        if points_coords is not None:
            assert (
                points_labels is not None
            ), "Expected points_labels to be provided if points_coords is provided, got None"
            assert (
                points_coords.ndim == 3
                and points_coords.shape[0] == batch_size
                and points_coords.shape[2] == 2
            ), f"Expected points_coords to be of shape (B, N, 2), got {points_coords.shape}"
            assert (
                points_labels.ndim == 2
                and points_labels.shape == points_coords.shape[:2]
            ), f"Expected points_labels to be of shape (B, N), got {points_labels.shape}"
            points_coords = self._transforms.transform_coords(
                points_coords, normalize=True, orig_hw=orig_hw
            )
            points = (points_coords, points_labels)

        # TODO: Add back if encoding prompts in batch
        # else:
        #     # If no points are provided, pad with an empty point (with label -1)
        #     points_coords = torch.zeros(batch_size, 1, 2, device=self.device)
        #     points_labels = -torch.ones(
        #         batch_size, 1, dtype=torch.int32, device=self.device
        #     )

        masks_low_res_logits = None

        if masks_logits is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert masks_logits.ndim == 4 and masks_logits.shape[:2] == (
                batch_size,
                1,
            ), f"Expected masks to be of shape (B, 1, H, W), got {masks_logits.shape}"

            masks_low_res_logits = self._transforms.downscale_masks_logits(
                masks_low_res_logits
            )

        if boxes is not None:
            assert (
                boxes.ndim == 3 and boxes.shape[0] == batch_size and boxes.shape[2] == 4
            ), f"Expected boxes to be of shape (B, N, 4), got {boxes.shape}"
            n_boxes = boxes.shape[1]
            # Encode the boxes as points with labels 2 and 3
            box_points_coords = boxes.reshape(batch_size, n_boxes * 2, 2)
            box_points_coords = self._transforms.transform_coords(
                box_points_coords, normalize=True, orig_hw=orig_hw
            )
            box_points_labels = torch.tensor(
                [2, 3], dtype=torch.int32, device=boxes.device
            ).repeat(batch_size, n_boxes)

            # Concatenate the box points with the existing points
            if points is not None:
                points = (
                    torch.cat([points[0], box_points_coords], dim=1),
                    torch.cat([points[1], box_points_labels], dim=1),
                )
            else:
                points = (box_points_coords, box_points_labels)

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder.forward(
            points=points, masks=masks_low_res_logits, boxes=None
        )

        return sparse_embeddings, dense_embeddings

    @torch.inference_mode()
    def generate_masks(
        self,
        orig_hw: tuple[int, int],
        img_embeddings: list[torch.Tensor],
        prompt_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        multimask_output: bool = True,
    ) -> SAM2Result:

        low_res_img_embeddings = img_embeddings[-1]
        high_res_img_embeddings = img_embeddings[:-1]

        if len(high_res_img_embeddings) == 0:
            high_res_img_embeddings = None

        if low_res_img_embeddings.ndim == 3:
            low_res_img_embeddings = low_res_img_embeddings.unsqueeze(0)

        if high_res_img_embeddings is not None and high_res_img_embeddings[0].ndim == 2:
            high_res_img_embeddings = [
                h.unsqueeze(0) for h in high_res_img_embeddings
            ]

        B, C, H, W = low_res_img_embeddings.shape

        assert C == self.sam_prompt_embed_dim
        assert H == self.sam_image_embedding_size
        assert W == self.sam_image_embedding_size

        if high_res_img_embeddings is not None:
            assert len(high_res_img_embeddings) == 2
            assert high_res_img_embeddings[0].shape == (B, C // 8, 4 * H, 4 * W)
            assert high_res_img_embeddings[1].shape == (B, C // 4, 2 * H, 2 * W)

        if prompt_embeddings is None:
            sparse_prompt_embeddings, dense_prompt_embeddings = (
                self.empty_prompt_embeddings
            )
            sparse_prompt_embeddings = sparse_prompt_embeddings.to(self.device)
            dense_prompt_embeddings = dense_prompt_embeddings.to(self.device)
        else:
            sparse_prompt_embeddings, dense_prompt_embeddings = prompt_embeddings

        prompt_positional_encoding = self.sam_prompt_encoder.get_dense_pe()

        (
            masks_logits,
            ious,
            sam_output_tokens,
            obj_scores_logits,
        ) = self.sam_mask_decoder.forward(
            image_embeddings=low_res_img_embeddings,
            image_pe=prompt_positional_encoding,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # TODO
            high_res_features=high_res_img_embeddings,
        )

        # Upscale the masks to the image_size
        masks_logits = self._transforms.postprocess_masks(
            masks_logits, (self.image_size, self.image_size)
        )
        masks_logits = torch.clamp(masks_logits, -32.0, 32.0)

        # Apply non-overlapping constraints if specified
        if self.non_overlap_masks:
            masks_logits = self._apply_non_overlapping_constraints(masks_logits)

        masks_logits = self._transforms.upscale_masks_logits(masks_logits, orig_hw)

        # Extract object pointer from the SAM output token (with occlusion handling)
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.obj_ptr_proj.forward(sam_output_token)

        # TODO: review this part. I'm not sure if this is correct.
        # Allow *soft* no obj ptr, unlike masks
        if self.soft_no_obj_ptr:
            obj_visibility = torch.sigmoid(obj_scores_logits)
        else:
            obj_visibility = (obj_scores_logits > 0).float()

        if self.fixed_no_obj_ptr:
            obj_ptr = obj_visibility * obj_ptr
        obj_ptr = obj_ptr + (1 - obj_visibility) * self.no_obj_ptr

        return SAM2Result(
            masks_logits=masks_logits,
            ious=ious,
            obj_ptrs=obj_ptr,
            obj_scores_logits=obj_scores_logits,
        )
