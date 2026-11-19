<script lang="ts">
  import { app } from '../lib/state.svelte';
  import CameraCell from './CameraCell.svelte';
</script>

<div class="grid">
  {#each app.segment?.cams ?? [] as cam (cam.cam)}
    <CameraCell {cam} />
  {/each}
</div>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--color-border);
  }

  @media (max-width: 800px) {
    .grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
