// Default VitePress theme plus our own colours, fonts and a custom 404.
import { h } from 'vue'
import DefaultTheme from 'vitepress/theme'
import NotFound from './NotFound.vue'
import DemoFrame from './DemoFrame.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'not-found': () => h(NotFound),
    })
  },
  // available in any markdown page, see how-it-works/scale-factors-demo.md
  enhanceApp({ app }) {
    app.component('DemoFrame', DemoFrame)
  },
}
