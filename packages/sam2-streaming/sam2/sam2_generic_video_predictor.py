from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch

from sam2.cuda_graphs import GraphCache
from sam2.modeling.sam2_generic import SAM2Generic
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.memory import ObjectMemory, ObjectMemorySelection
from sam2.modeling.sam2_memory import ObjectMemoryBank, SAM2ObjectMemoryBank


@dataclass
class SAM2GenericVideoPredictorState:
    video_hw: tuple[int, int]
    memory_bank: ObjectMemoryBank = field(default_factory=SAM2ObjectMemoryBank)

    def __post_init__(self):
        if self.memory_bank is None:
            raise ValueError("Memory bank cannot be None")
        if self.video_hw is None:
            raise ValueError("Video height and width cannot be None")

    @staticmethod
    def create(
        video_hw: tuple[int, int], memory_bank: ObjectMemoryBank | None = None
    ) -> "SAM2GenericVideoPredictorState":
        if memory_bank is None:
            memory_bank = SAM2ObjectMemoryBank()
        return SAM2GenericVideoPredictorState(
            video_hw=video_hw, memory_bank=memory_bank
        )


class SAM2GenericVideoPredictor(SAM2Generic):
    """
    SAM2GenericVideoPredictor provides a handy video prediction interface.

    Note: works in a forward-only manner.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._device_indices: dict[tuple, torch.Tensor] = {}
        """Index tensors on the device by value, kept for the life of the predictor: a CUDA graph reads them in place."""

    @torch.inference_mode()
    def forward(
        self,
        state: SAM2GenericVideoPredictorState,
        frame_idx: int,
        frame: torch.Tensor,
        prompts: list[SAM2Prompt] = [],
        multimask_output: bool = True,
        reverse_tracking: bool = False,
        create_memory: bool = True,
    ) -> dict[int, SAM2Result]:
        assert frame.shape in [
            (1, *state.video_hw),
            (3, *state.video_hw),
        ], f"Expected frame to be of shape (C, H, W) or (1, C, H, W) with H and W equal to {state.video_hw}, got {frame.shape}"

        img_embeddings, img_pos_embeddings = self.encode_image(frame)

        return self.forward_embeddings(
            state=state,
            frame_idx=frame_idx,
            img_embeddings=img_embeddings,
            img_pos_embeddings=img_pos_embeddings,
            prompts=prompts,
            multimask_output=multimask_output,
            reverse_tracking=reverse_tracking,
            create_memory=create_memory,
        )

    @torch.inference_mode()
    def forward_embeddings(
        self,
        state: SAM2GenericVideoPredictorState,
        frame_idx: int,
        img_embeddings: list[torch.Tensor],
        img_pos_embeddings: list[torch.Tensor],
        prompts: list[SAM2Prompt] = [],
        multimask_output: bool = True,
        reverse_tracking: bool = False,
        create_memory: bool = True,
    ) -> dict[int, SAM2Result]:

        assert prompts is None or np.unique([p.obj_id for p in prompts]).size == len(
            prompts
        ), "Only one prompt per object should be provided"

        # Unique list of all objects to propagate the masks for (includes previous objects and new prompts).
        all_obj_ids = state.memory_bank.known_obj_ids | set([p.obj_id for p in prompts])
        n_objs = len(all_obj_ids)

        prompts_dicts: dict[int, SAM2Prompt] = {
            prompt.obj_id: prompt for prompt in prompts
        }

        objects_selected_memories = state.memory_bank.select_memories(
            obj_ids=state.memory_bank.known_obj_ids,
            current_frame_idx=frame_idx,
            max_conditional_memories=self.max_cond_frames_in_attn,
            max_non_conditional_memories=self.num_maskmem - 1,
            max_ptr_memories=self.max_obj_ptrs_in_encoder,
            only_include_pointers_in_past=self.only_obj_ptrs_in_the_past_for_eval,
            reverse_tracking=reverse_tracking,
        )

        results: list[SAM2Result] = []

        for obj_id in all_obj_ids:
            prompt = prompts_dicts.get(obj_id, None)
            has_prompt = prompt is not None

            if has_prompt:
                prompt = prompt.to(self.device)

                point_coords = (
                    prompt.points_coords.unsqueeze(0)
                    if prompt.points_coords is not None
                    else None
                )
                point_labels = (
                    prompt.points_labels.unsqueeze(0)
                    if prompt.points_labels is not None
                    else None
                )
                boxes = prompt.boxes.unsqueeze(0) if prompt.boxes is not None else None
                masks_logits = (
                    prompt.masks_logits.unsqueeze(0)
                    if prompt.masks_logits is not None
                    else None
                )

                prompt_embeddings = self.encode_prompts(
                    orig_hw=state.video_hw,
                    points_coords=point_coords,
                    points_labels=point_labels,
                    boxes=boxes,
                    masks_logits=masks_logits,
                )

                result = self.generate_masks(
                    orig_hw=state.video_hw,
                    img_embeddings=img_embeddings,
                    prompt_embeddings=prompt_embeddings,
                    multimask_output=multimask_output,
                )

            else:
                assert (
                    obj_id in objects_selected_memories
                ), f"Expected memory bank to have a memory for object {obj_id} but it does not."

                object_selected_memories = objects_selected_memories[obj_id]
                # Transfer the memories to the correct device
                object_selected_memories = object_selected_memories.to(self.device)

                # Edge case if we forward a frame without any memories.
                if (
                    len(object_selected_memories.conditional_memories) == 0
                    and len(object_selected_memories.non_conditional_memories) == 0
                ):
                    continue

                conditioned_img_embeddings = self.condition_image_embeddings_on_memories(
                    frame_idx=frame_idx,
                    img_embeddings=img_embeddings,
                    img_pos_embeddings=img_pos_embeddings,
                    non_conditional_memories=object_selected_memories.non_conditional_memories,
                    conditional_memories=object_selected_memories.conditional_memories,
                    ptr_memories=object_selected_memories.ptr_memories,
                )

                result = self.generate_masks(
                    orig_hw=state.video_hw,
                    img_embeddings=conditioned_img_embeddings,
                    multimask_output=True,
                )

            results.append(result)

        # Edge case if we forward a frame without any prompts or memories.
        if len(results) == 0:
            return {}

        batched_results = SAM2Result.cat(results)

        if create_memory:
            is_prompt = torch.tensor(
                [obj_id in prompts_dicts for obj_id in all_obj_ids],
                dtype=torch.bool,
                device=batched_results.device,
            )

            memory_embeddings, memory_pos_embeddings = self.encode_memory(
                img_embeddings=[m.expand((n_objs, -1, -1, -1)) for m in img_embeddings],
                masks_logits=batched_results.best_mask_logits,
                obj_score_logits=batched_results.obj_score_logits,
                is_prompt=is_prompt,
            )

            state.memory_bank.try_add_memories(
                frame_idx=frame_idx,
                obj_ids=all_obj_ids,
                memory_embeddings=memory_embeddings,
                memory_pos_embeddings=memory_pos_embeddings,
                results=batched_results,
                prompts=prompts,
            )

            state.memory_bank.prune_memories(
                obj_ids=all_obj_ids,
                current_frame_idx=frame_idx,
            )

        return {obj_id: result for obj_id, result in zip(all_obj_ids, batched_results)}

    def index_tensor(self, values: list, dtype: torch.dtype = torch.long) -> torch.Tensor:
        """``values`` on the device without a host sync: a copy from pageable host memory waits for all queued GPU work,
        one from pinned memory with ``non_blocking`` does not."""
        if self.device.type != "cuda":
            return torch.tensor(values, dtype=dtype, device=self.device)
        return torch.tensor(values, dtype=dtype, pin_memory=True).to(self.device, non_blocking=True)

    @torch.inference_mode()
    def forward_batched(
        self,
        states: list[SAM2GenericVideoPredictorState],
        frame_idx: int,
        frames: torch.Tensor,
        prompts: list[list[SAM2Prompt]],
        multimask_output: list[bool],
        graphs: GraphCache | None = None,
        best_only: bool = True,
        one_graph: bool = False,
    ) -> list[dict[int, SAM2Result]]:
        """``forward`` for several videos at the same timestep, each module run once per group instead of per object.

        One image-encoder call for all ``frames`` (B, C, H, W), one per video state. Prompted objects are grouped by
        prompt layout, propagated objects by memory layout (numbers of conditional, non-conditional and pointer
        memories); each group is one batch through memory attention and the mask decoder, and one memory-encoder call
        covers every object. Per object the operations are those of ``forward`` (each pointer memory keeps its own
        frame index); only batch-size-dependent kernel numerics differ. With ``best_only`` the masks carry the best
        hypothesis only; without, every hypothesis the mask decoder makes, in its order (three for propagated objects,
        as ``forward``), and the memories still encode the best one.
        With ``graphs``, the image encoder, memory attention, mask decoder and memory encoder replay CUDA graphs per
        input signature (same kernels and layouts, so the same results); the selection and batching glue stays eager.
        With ``graphs`` and ``one_graph``, a step without prompts is one graph per signature instead: the image
        encoder, every memory-layout group, the memory encoder and the batching between them (the memories enter
        stacked, one tensor per kind and group); only the memory selection and the banks' bookkeeping stay eager.

        Returns:
            Per state, the results by object id.
        """
        assert len(states) == len(prompts) == len(multimask_output) == frames.shape[0]
        # a batch row is one object of one video: non-overlap constraints across rows would let videos suppress each other
        assert not self.non_overlap_masks_for_mem_enc and not self.non_overlap_masks, "the batch mixes videos"
        video_hw: tuple[int, int] = states[0].video_hw
        assert all(s.video_hw == video_hw for s in states) and tuple(frames.shape[-2:]) == video_hw
        if graphs is not None:
            assert not self.add_tpos_enc_to_obj_ptrs, "pointer temporal encodings copy frame indices to the device: no graph capture"

        if graphs is not None and one_graph and not any(prompts):
            items = self._collect_items(states, frame_idx, prompts)
            if not items:
                return [{} for _ in states]
            results, memory_embeddings, memory_pos_embeddings = self._propagate_in_one_graph(items, frames, video_hw, graphs, best_only)
            return self._add_memories(states, frame_idx, prompts, items, results, memory_embeddings, memory_pos_embeddings)

        img_embeddings, img_pos_embeddings = _run(graphs, ("encode_image",), self.encode_image, [frames])
        items: list[_Item] = self._collect_items(states, frame_idx, prompts)
        groups: dict[_PromptLayout | _MemoryLayout, list[int]] = {}
        for index, item in enumerate(items):
            layout = _PromptLayout.of(item.prompt, multimask_output[item.slot]) if item.prompt is not None else _MemoryLayout.of(item.selection)
            groups.setdefault(layout, []).append(index)

        by_index: dict[int, SAM2Result] = {}
        for layout, members in groups.items():
            group: list[_Item] = [items[i] for i in members]
            slots = self.index_tensor([item.slot for item in group])
            group_img = [level.index_select(0, slots) for level in img_embeddings]
            if isinstance(layout, _PromptLayout):
                result = self._decode_prompted(layout, group, group_img, video_hw, graphs, best_only)
            else:
                group_low_pos = img_pos_embeddings[-1].index_select(0, slots)
                result = self._propagate(layout, group, group_img, group_low_pos, frame_idx, video_hw, graphs, best_only)
            for row, index in enumerate(members):
                by_index[index] = result[row]

        if not items:
            return [{} for _ in states]
        results: list[SAM2Result] = [by_index[i] for i in range(len(items))]
        return self._store_memories(states, frame_idx, prompts, items, results, img_embeddings[-1], graphs)

    def _collect_items(
        self, states: list[SAM2GenericVideoPredictorState], frame_idx: int, prompts: list[list[SAM2Prompt]]
    ) -> list["_Item"]:
        """The objects of this step in ``forward``'s per-state order: prompted, or propagated from their memories
        (objects without spatial memories are left out, as ``forward`` skips them)."""
        items: list[_Item] = []
        device: torch.device = self.device
        for slot, state in enumerate(states):
            slot_prompts: dict[int, SAM2Prompt] = {p.obj_id: p for p in prompts[slot]}
            assert len(slot_prompts) == len(prompts[slot]), "Only one prompt per object should be provided"
            selected = state.memory_bank.select_memories(
                obj_ids=state.memory_bank.known_obj_ids,
                current_frame_idx=frame_idx,
                max_conditional_memories=self.max_cond_frames_in_attn,
                max_non_conditional_memories=self.num_maskmem - 1,
                max_ptr_memories=self.max_obj_ptrs_in_encoder,
                only_include_pointers_in_past=self.only_obj_ptrs_in_the_past_for_eval,
                reverse_tracking=False,
            )
            for obj_id in state.memory_bank.known_obj_ids | set(slot_prompts):
                prompt: SAM2Prompt | None = slot_prompts.get(obj_id)
                if prompt is not None:
                    items.append(_Item(slot, obj_id, prompt.to(device), None))
                    continue
                selection = selected[obj_id]
                if len(selection.conditional_memories) == 0 and len(selection.non_conditional_memories) == 0:
                    continue
                # a bank stored on the device needs no copy (ObjectMemory.to rebuilds every memory, ~3 calls each)
                spatial = selection.conditional_memories + selection.non_conditional_memories
                items.append(_Item(slot, obj_id, None, selection if spatial[0].memory_embeddings.device == device else selection.to(device)))
        return items

    def _decode_prompted(
        self,
        layout: "_PromptLayout",
        group: list["_Item"],
        img_embeddings: list[torch.Tensor],
        video_hw: tuple[int, int],
        graphs: GraphCache | None,
        best_only: bool,
    ) -> SAM2Result:
        """Masks of a group of prompted objects (one prompt layout): one prompt-encoder and one mask-decoder batch."""
        group_prompts: list[SAM2Prompt] = [item.prompt for item in group]
        # One layout per group, so each field is None for all of them or stacks to (B, ...).
        fields: dict[str, torch.Tensor | None] = {
            name: None if getattr(group_prompts[0], name) is None else torch.stack([getattr(p, name) for p in group_prompts])
            for name in ("points_coords", "points_labels", "boxes", "masks_logits")
        }
        prompt_embeddings = self.encode_prompts(orig_hw=video_hw, batch_size=len(group), **fields)
        n_levels: int = len(img_embeddings)

        def decode(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return _decoded(
                self.generate_masks(
                    orig_hw=video_hw,
                    img_embeddings=list(tensors[:n_levels]),
                    prompt_embeddings=(tensors[n_levels], tensors[n_levels + 1]),
                    multimask_output=layout.multimask,
                    best_only=best_only,
                )
            )

        return SAM2Result(*_run(graphs, ("prompted", video_hw, layout.multimask, best_only), decode, [*img_embeddings, *prompt_embeddings]))

    def _propagate(
        self,
        layout: "_MemoryLayout",
        group: list["_Item"],
        img_embeddings: list[torch.Tensor],
        low_res_img_pos_embeddings: torch.Tensor,
        frame_idx: int,
        video_hw: tuple[int, int],
        graphs: GraphCache | None,
        best_only: bool,
    ) -> SAM2Result:
        """Masks of a group of propagated objects (one memory layout): memory k of every object stacks into one
        (B, ...) tensor, then one memory-attention and one mask-decoder batch."""
        selections: list[ObjectMemorySelection] = [item.selection for item in group]
        spatial: list[list[ObjectMemory]] = [[s.conditional_memories[k] for s in selections] for k in range(layout.n_cond)]
        spatial += [[s.non_conditional_memories[k] for s in selections] for k in range(layout.n_non_cond)]
        memory_embeddings = [torch.cat([m.memory_embeddings for m in per_object]) for per_object in spatial]
        memory_pos_embeddings = [torch.cat([m.memory_pos_embeddings for m in per_object]) for per_object in spatial]
        ptrs = [torch.cat([s.ptr_memories[k].ptr for s in selections]) for k in range(layout.n_ptr)]
        # The pointer frames per object, (n_ptr, B), only where pointer temporal encodings read them (never in a graph).
        ptr_frame_idx = (
            [self.index_tensor([[s.ptr_memories[k].frame_idx for s in selections] for k in range(layout.n_ptr)]).view(layout.n_ptr, len(group))]
            if self.add_tpos_enc_to_obj_ptrs
            else []
        )
        n_spatial: int = layout.n_cond + layout.n_non_cond

        def condition(low: torch.Tensor, low_pos: torch.Tensor, *memories: torch.Tensor) -> torch.Tensor:
            return self.condition_on_memory_tensors(
                frame_idx=frame_idx,
                low_res_img_embeddings=low,
                low_res_img_pos_embeddings=low_pos,
                memory_embeddings=list(memories[:n_spatial]),
                memory_pos_embeddings=list(memories[n_spatial : 2 * n_spatial]),
                n_conditional=layout.n_cond,
                ptrs=list(memories[2 * n_spatial : 2 * n_spatial + layout.n_ptr]),
                ptr_frame_idx=memories[-1] if ptr_frame_idx else None,
            )

        conditioned_low = _run(
            graphs,
            ("condition", layout),
            condition,
            [img_embeddings[-1], low_res_img_pos_embeddings, *memory_embeddings, *memory_pos_embeddings, *ptrs, *ptr_frame_idx],
        )
        empty_sparse, empty_dense = self.empty_prompt_embeddings

        def decode(*levels: torch.Tensor) -> tuple[torch.Tensor, ...]:
            n: int = levels[0].shape[0]
            return _decoded(
                self.generate_masks(
                    orig_hw=video_hw,
                    img_embeddings=list(levels),
                    prompt_embeddings=(empty_sparse.expand(n, -1, -1), empty_dense.expand(n, -1, -1, -1)),
                    multimask_output=True,
                    best_only=best_only,
                )
            )

        return SAM2Result(*_run(graphs, ("propagated", video_hw, best_only), decode, [*img_embeddings[:-1], conditioned_low]))

    def _store_memories(
        self,
        states: list[SAM2GenericVideoPredictorState],
        frame_idx: int,
        prompts: list[list[SAM2Prompt]],
        items: list["_Item"],
        results: list[SAM2Result],
        low_res_img_embeddings: torch.Tensor,
        graphs: GraphCache | None,
    ) -> list[dict[int, SAM2Result]]:
        """One memory-encoder call for every object, then each state's bank stores its own rows."""
        all_results = SAM2Result.cat(results)  # objects with different hypothesis counts are reduced to the best one
        best_masks = all_results.masks_logits if all_results.masks_logits.shape[1] == 1 else all_results.best_mask_logits
        slots = self.index_tensor([item.slot for item in items])

        def encode(low: torch.Tensor, masks_logits: torch.Tensor, obj_score_logits: torch.Tensor, is_prompt: torch.Tensor):
            return self.encode_memory(img_embeddings=[low], masks_logits=masks_logits, obj_score_logits=obj_score_logits, is_prompt=is_prompt)

        memory_embeddings, memory_pos_embeddings = _run(
            graphs,
            ("encode_memory",),
            encode,
            [
                low_res_img_embeddings.index_select(0, slots),
                best_masks,
                all_results.obj_score_logits,
                self.index_tensor([item.prompt is not None for item in items], torch.bool),
            ],
        )
        return self._add_memories(states, frame_idx, prompts, items, results, memory_embeddings, memory_pos_embeddings)

    def _add_memories(
        self,
        states: list[SAM2GenericVideoPredictorState],
        frame_idx: int,
        prompts: list[list[SAM2Prompt]],
        items: list["_Item"],
        results: list[SAM2Result],
        memory_embeddings: torch.Tensor,
        memory_pos_embeddings: torch.Tensor,
    ) -> list[dict[int, SAM2Result]]:
        """Each state's bank stores its own rows of the step's memories; the results by object id per state."""
        outputs: list[dict[int, SAM2Result]] = [{} for _ in states]
        for slot, state in enumerate(states):
            rows = [i for i, item in enumerate(items) if item.slot == slot]
            if not rows:
                continue
            obj_ids = [items[i].obj_id for i in rows]
            # items are in state order, so a state's rows are one range: views, no copies (the bank copies its rows)
            first, end = rows[0], rows[-1] + 1
            assert rows == list(range(first, end))
            state.memory_bank.try_add_memories(
                frame_idx=frame_idx,
                obj_ids=obj_ids,
                memory_embeddings=memory_embeddings[first:end],
                memory_pos_embeddings=memory_pos_embeddings[first:end],
                results=results[first] if len(rows) == 1 else SAM2Result.cat([results[i] for i in rows]),
                prompts=prompts[slot],
            )
            state.memory_bank.prune_memories(obj_ids=obj_ids, current_frame_idx=frame_idx)
            outputs[slot] = {obj_ids[k]: results[i] for k, i in enumerate(rows)}
        return outputs

    def _propagate_in_one_graph(
        self,
        items: list["_Item"],
        frames: torch.Tensor,
        video_hw: tuple[int, int],
        graphs: GraphCache,
        best_only: bool,
    ) -> tuple[list[SAM2Result], torch.Tensor, torch.Tensor]:
        """A step of propagated objects only, as one CUDA graph per signature: per memory-layout group the operations of
        ``_propagate``, then ``_store_memories``' memory encoding, on ``encode_image(frames)``.

        Returns:
            The results in item order and the memory embeddings and positions, one row per item.
        """
        groups: dict[_MemoryLayout, list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault(_MemoryLayout.of(item.selection), []).append(index)
        layouts: list[_MemoryLayout] = list(groups)
        slots: list[torch.Tensor] = [self._device_index([items[i].slot for i in members]) for members in groups.values()]
        all_slots: torch.Tensor = self._device_index([item.slot for item in items])
        order: list[int] = [i for members in groups.values() for i in members]
        # rows of the concatenated group outputs back to item order (None when the groups already are in item order)
        to_items: torch.Tensor | None = None if order == list(range(len(items))) else self._device_index([order.index(i) for i in range(len(items))])
        inputs: list[torch.Tensor] = [frames]
        for layout, members in groups.items():
            selections: list[ObjectMemorySelection] = [items[i].selection for i in members]
            spatial = [m for k in range(layout.n_cond) for m in (s.conditional_memories[k] for s in selections)]
            spatial += [m for k in range(layout.n_non_cond) for m in (s.non_conditional_memories[k] for s in selections)]
            # memory k of the group's objects is rows [k * B, (k + 1) * B) of one stacked tensor per kind
            inputs += [torch.cat([m.memory_embeddings for m in spatial]), torch.cat([m.memory_pos_embeddings for m in spatial])]
            if layout.n_ptr:
                inputs.append(torch.cat([s.ptr_memories[k].ptr for k in range(layout.n_ptr) for s in selections]))
        empty_sparse, empty_dense = self.empty_prompt_embeddings

        def step(frames: torch.Tensor, *memories: torch.Tensor) -> tuple[torch.Tensor, ...]:
            img_embeddings, img_pos_embeddings = self.encode_image(frames)
            stacked = iter(memories)
            decoded: list[tuple[torch.Tensor, ...]] = []
            for layout, index in zip(layouts, slots, strict=True):
                n: int = index.shape[0]
                memory_embeddings, memory_pos_embeddings = next(stacked).split(n), next(stacked).split(n)
                ptrs = list(next(stacked).split(n)) if layout.n_ptr else []
                group_img = [level.index_select(0, index) for level in img_embeddings]
                conditioned_low = self.condition_on_memory_tensors(
                    frame_idx=0,  # read only with pointer temporal encodings, which graphs refuse
                    low_res_img_embeddings=group_img[-1],
                    low_res_img_pos_embeddings=img_pos_embeddings[-1].index_select(0, index),
                    memory_embeddings=list(memory_embeddings),
                    memory_pos_embeddings=list(memory_pos_embeddings),
                    n_conditional=layout.n_cond,
                    ptrs=ptrs,
                    ptr_frame_idx=None,
                )
                result = self.generate_masks(
                    orig_hw=video_hw,
                    img_embeddings=[*group_img[:-1], conditioned_low],
                    prompt_embeddings=(empty_sparse.expand(n, -1, -1), empty_dense.expand(n, -1, -1, -1)),
                    multimask_output=True,
                    best_only=best_only,
                )
                decoded.append(_decoded(result))
            group_results = [SAM2Result(*d) for d in decoded]
            best_masks = torch.cat([r.masks_logits if r.masks_logits.shape[1] == 1 else r.best_mask_logits for r in group_results])
            obj_score_logits = torch.cat([r.obj_score_logits for r in group_results])
            if to_items is not None:
                best_masks, obj_score_logits = best_masks.index_select(0, to_items), obj_score_logits.index_select(0, to_items)
            memory_embeddings, memory_pos_embeddings = self.encode_memory(
                img_embeddings=[img_embeddings[-1].index_select(0, all_slots)],
                masks_logits=best_masks,
                obj_score_logits=obj_score_logits,
                is_prompt=torch.zeros(len(items), dtype=torch.bool, device=frames.device),
            )
            return (*(t for d in decoded for t in d), memory_embeddings, memory_pos_embeddings)

        key = ("step", video_hw, best_only, tuple(layouts), tuple(tuple(items[i].slot for i in members) for members in groups.values()), tuple(order))
        outputs = graphs(key, step, inputs)
        by_index: dict[int, SAM2Result] = {}
        for g, members in enumerate(groups.values()):
            result = SAM2Result(*outputs[4 * g : 4 * g + 4])
            for row, index in enumerate(members):
                by_index[index] = result[row]
        return [by_index[i] for i in range(len(items))], outputs[-2], outputs[-1]

    def _device_index(self, values: list[int]) -> torch.Tensor:
        """``values`` as a long tensor on the device, made once per value list (a host sync the first time only)."""
        key = tuple(values)
        index: torch.Tensor | None = self._device_indices.get(key)
        if index is None:
            index = self._device_indices[key] = torch.tensor(values, dtype=torch.long, device=self.device)
        return index


@dataclass(frozen=True, slots=True)
class _Item:
    """One object of one video at a ``forward_batched`` step: prompted (``prompt``) or propagated (``selection``)."""

    slot: int
    obj_id: int
    prompt: SAM2Prompt | None
    selection: ObjectMemorySelection | None


class _PromptLayout(NamedTuple):
    """What a batch of prompted objects shares: the mask count and the shapes of the prompt tensors."""

    multimask: bool
    points: tuple[int, ...] | None
    boxes: tuple[int, ...] | None
    masks: tuple[int, ...] | None

    @staticmethod
    def of(prompt: SAM2Prompt, multimask: bool) -> "_PromptLayout":
        def shape(t: torch.Tensor | None) -> tuple[int, ...] | None:
            return None if t is None else tuple(t.shape)

        return _PromptLayout(multimask, shape(prompt.points_coords), shape(prompt.boxes), shape(prompt.masks_logits))


class _MemoryLayout(NamedTuple):
    """What a batch of propagated objects shares: the numbers of conditional, non-conditional and pointer memories."""

    n_cond: int
    n_non_cond: int
    n_ptr: int

    @staticmethod
    def of(selection: ObjectMemorySelection) -> "_MemoryLayout":
        return _MemoryLayout(len(selection.conditional_memories), len(selection.non_conditional_memories), len(selection.ptr_memories))


def _run(graphs: GraphCache | None, key: tuple, fn: Callable[..., Any], inputs: list[torch.Tensor]) -> Any:
    """``fn(*inputs)``, eager or from ``graphs``' CUDA graph of this signature."""
    return fn(*inputs) if graphs is None else graphs(key, fn, inputs)


def _decoded(result: SAM2Result) -> tuple[torch.Tensor, ...]:
    """A mask decoder result as graph outputs (tensors only), in ``SAM2Result``'s field order."""
    return result.masks_logits, result.ious, result.obj_ptrs, result.obj_score_logits
