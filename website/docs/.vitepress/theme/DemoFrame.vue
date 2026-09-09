<!--
  Wraps a standalone demo page from public/ in an iframe. The iframe keeps the demo's
  own CSS away from the docs theme, but it also means the frame has no idea how tall
  the demo is, so the demo posts its height back and we resize to match. The dark mode
  toggle is passed in the same way, otherwise a light demo sits in a dark page.
-->
<script setup>
import { ref, watch, onMounted, onBeforeUnmount } from 'vue'
import { withBase, useData } from 'vitepress'

const props = defineProps({
  src: { type: String, required: true },
  title: { type: String, default: 'demo' },
  min: { type: Number, default: 700 },
  // run at the demo's own width instead of filling the column
  wide: { type: Boolean, default: false },
})

// how much clear space the wide demo leaves to the right of the window
const GUTTER = 96
const NATURAL = 1025

const { isDark } = useData()
const frame = ref(null)
const height = ref(props.min)
const width = ref(NATURAL)

// A wide demo runs at its own width, but never past the right of the window.
// Without this it keeps its 1025px on a narrow screen and pushes off the edge.
function fit() {
  if (!props.wide || !frame.value) return
  const left = frame.value.getBoundingClientRect().left
  width.value = Math.max(320, Math.min(NATURAL, window.innerWidth - left - GUTTER))
}

function tell() {
  if (frame.value && frame.value.contentWindow) {
    frame.value.contentWindow.postMessage({ theme: isDark.value ? 'dark' : 'light' }, '*')
  }
}

function onMessage(ev) {
  if (!frame.value || ev.source !== frame.value.contentWindow) return
  const h = ev.data && ev.data.demoHeight
  if (typeof h === 'number') height.value = Math.max(props.min, Math.ceil(h))
}

watch(isDark, tell)
onMounted(() => {
  window.addEventListener('message', onMessage)
  window.addEventListener('resize', fit)
  fit()
  // the iframe may well have loaded before we hydrated, in which case its first
  // height went nowhere, so ask for it again
  tell()
})
onBeforeUnmount(() => {
  window.removeEventListener('message', onMessage)
  window.removeEventListener('resize', fit)
})
</script>

<template>
  <iframe
    ref="frame"
    class="demoframe"
    :src="withBase(src)"
    :title="title"
    :style="{ height: height + 'px', width: wide ? width + 'px' : null }"
    :class="{ wide }"
    @load="() => { fit(); tell() }"
  />
</template>

<style>
.demoframe {
  width: 100%;
  border: 0;
  display: block;
  margin: 4px 0;
  background: transparent;
}
.demoframe.wide {
  max-width: none;
}
</style>
