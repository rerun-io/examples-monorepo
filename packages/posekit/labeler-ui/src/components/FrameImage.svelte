<script lang="ts">
  import { loadFrame } from '../lib/frames';

  interface Props {
    url: string;
    t: number;
    alt: string;
    /** Grid time of the frame on screen; the boxes are drawn for this time, so they always match the image. */
    shownT?: number | null;
  }

  let { url, t, alt, shownT = $bindable(null) }: Props = $props();
  let src = $state<string | null>(null);

  // The newest request wins: a frame that arrives after a later one was asked for is never shown (playback skips it).
  $effect(() => {
    const wantT = t;
    let live = true;
    loadFrame(url)
      .then(async (objectUrl) => {
        const img = new Image();
        img.src = objectUrl;
        await img.decode();
        if (!live) return;
        src = objectUrl;
        shownT = wantT;
      })
      .catch(() => {
        if (!live) return;
        src = null;
        shownT = wantT;
      });
    return () => {
      live = false;
    };
  });
</script>

{#if src}
  <img {src} {alt} draggable="false" />
{/if}

<style>
  img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    user-select: none;
  }
</style>
