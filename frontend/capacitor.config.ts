import type { CapacitorConfig } from '@capacitor/cli'

// Local-Wi-Fi debug builds talk to an http:// FastAPI backend. In that case the
// WebView must also be served over http:// (otherwise https://localhost -> http://
// is blocked as mixed content). Set CAP_DEBUG_HTTP=1 only for those debug builds.
// Release builds should use an https:// backend and leave this unset.
const debugHttp = process.env.CAP_DEBUG_HTTP === '1'

const config: CapacitorConfig = {
  appId: 'com.passportseva.guide',
  appName: 'Passport Seva Application Guide',
  webDir: 'dist',
  server: debugHttp ? { androidScheme: 'http', cleartext: true } : {},
}

export default config

