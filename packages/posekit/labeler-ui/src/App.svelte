<script lang="ts">
  import { onMount } from 'svelte';
  import { app } from './lib/state.svelte';
  import CameraGrid from './components/CameraGrid.svelte';
  import FixView from './components/FixView.svelte';
  import Header from './components/Header.svelte';
  import KeysRow from './components/KeysRow.svelte';
  import SolveStrip from './components/SolveStrip.svelte';
  import TransportBar from './components/TransportBar.svelte';

  const fixCam = $derived(app.segment?.cams.find((c) => c.cam === app.fixCam));

  onMount(() => {
    const onhashchange = () => app.load(location.hash);
    onhashchange();
    window.addEventListener('hashchange', onhashchange);
    return () => window.removeEventListener('hashchange', onhashchange);
  });

  // Keep the next frames' JPEGs coming while the playhead moves.
  $effect(() => app.prefetch());

  function onkeydown(e: KeyboardEvent): void {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const act: Record<string, () => void> = {
      ' ': () => app.togglePlay(),
      ArrowLeft: () => app.step(-1),
      ArrowRight: () => app.step(1),
      a: () => app.accept(),
      l: () => app.setSide('left'),
      r: () => app.setSide('right'),
      Delete: () => app.deleteSelected(),
      Backspace: () => app.deleteSelected(),
      Enter: () => app.save(),
      Escape: () => app.back()
    };
    const digit = /^[1-6]$/.test(key) ? () => app.openFix(Number(key) - 1) : undefined;
    const action = act[key] ?? digit;
    if (!action) return;
    e.preventDefault();
    // A focused button would also take Space/Enter as a click.
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    action();
  }
</script>

<svelte:window {onkeydown} />

<main>
  <Header />
  {#if app.segment}
    {#if fixCam}
      <FixView cam={fixCam} />
    {:else}
      <CameraGrid />
    {/if}
    <SolveStrip />
    <TransportBar />
  {:else}
    <p class="empty">{app.message ? '' : 'Loading…'}</p>
  {/if}
  <KeysRow />
</main>

<style>
  main {
    max-width: 1400px;
    margin: 16px auto 32px;
    border-top: 1px solid var(--color-border);
    border-bottom: 1px solid var(--color-border);
  }

  .empty {
    padding: 40px 16px;
    margin: 0;
    color: var(--color-muted);
  }
</style>
