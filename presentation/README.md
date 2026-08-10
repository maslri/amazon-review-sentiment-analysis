# Amazon Review Intelligence — presentation site

Public, responsive project presentation for the Amazon review rating prediction repository.

Live site: [amazon-review-intelligence.ma-slri2128.chatgpt.site](https://amazon-review-intelligence.ma-slri2128.chatgpt.site)

## Local development

Requires Node.js `>=22.13.0`.

```bash
npm install
npm run dev
```

Validation and production build:

```bash
npm run lint
npm run build
```

The site is implemented with React, TypeScript, vinext and plain CSS. Deployment identity is stored in `.openai/hosting.json`; generated build outputs remain ignored.
