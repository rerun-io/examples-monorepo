<script lang="ts">
  import { fmtSeconds, shortSegment } from '../lib/clip';
  import { app } from '../lib/state.svelte';

  const clip = $derived(
    app.segment
      ? `${shortSegment(app.sid)} · clip ${app.seconds(app.times[app.i0]).toFixed(1)}–${fmtSeconds(app.seconds(app.times[app.i1]))}`
      : ''
  );
  const acceptedTo = $derived(app.accepted > 0 ? app.times[app.i0 + app.accepted - 1] : undefined);
</script>

<header>
  <span class="label title">Hand labeler</span>
  <span class="clip">{clip}</span>
  <span class="legend label">
    <span class="left"><i></i>left</span>
    <span class="right"><i></i>right</span>
    <span><i></i>labeled</span>
    <span class="derived"><i></i>derived (3D)</span>
  </span>
  {#if app.message}
    <span class="message {app.message.tone}" role="status">{app.message.text}</span>
  {/if}
  <span class="progress label">
    {acceptedTo === undefined
      ? 'Nothing accepted'
      : `Accepted to ${fmtSeconds(app.seconds(acceptedTo))}`}
    · {app.clipFixes}
    {app.clipFixes === 1 ? 'fix' : 'fixes'}
  </span>
  <button class="btn" onclick={() => app.commit()} disabled={app.busy || !app.segment}>
    Write layer
  </button>
</header>

<style>
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 20px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-border);
  }

  .title {
    font-size: 13px;
    color: var(--color-strong);
  }

  .clip {
    color: var(--color-muted);
  }

  .legend {
    display: flex;
    gap: 18px;
    font-size: 12px;
    font-weight: 500;
    color: var(--color-muted);

    & i {
      display: inline-block;
      width: 18px;
      height: 0;
      border-top: 3px solid var(--color-strong);
      vertical-align: middle;
      margin-right: 6px;
    }

    & .left {
      color: var(--hand-left);

      & i {
        border-color: var(--hand-left);
      }
    }

    & .right {
      color: var(--hand-right);

      & i {
        border-color: var(--hand-right);
      }
    }

    & .derived i {
      border-top-style: dashed;
    }
  }

  .message {
    flex: 1 1 0;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: right;
    font-size: 0.9rem;
    color: var(--color-text);

    &.good {
      color: var(--color-success);
    }

    &.warn {
      color: var(--color-warning);
    }

    &.bad {
      color: var(--color-danger);
    }
  }

  .progress {
    margin-left: auto;
    font-size: 13px;
    font-weight: 500;
    color: var(--color-success);
  }

  .message + .progress {
    margin-left: 0;
  }

  .btn {
    padding: 6px 10px;
    font-size: 12px;
  }
</style>
