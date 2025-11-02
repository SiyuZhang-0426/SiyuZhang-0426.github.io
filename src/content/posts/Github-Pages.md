---
title: Github-Pages
published: 2025-10-30
description: How to build Github Pages
tags: [Github, Guide]
category: Guide
draft: false
---

本项目源地址：[saicaca fuwari](https://github.com/saicaca/fuwari?tab=readme-ov-file)。

# Background

Github Pages服务，能够将托管在名为siyuzhang-0426.github.io仓库下的前端代码渲染为静态页面。在部署后，可以通过[https://www.siyuzhang-0426.github.io](https://www.siyuzhang-0426.github.io)访问。

项目主要使用到Astro，一个现代化的静态网站生成器，专为构建快速、内容驱动的网站而设计。

# Preparation

首先，我们需要准备如下的工具：Node.js、npm、pnpm、Git。这里具体准备方法不再赘述。

随后，在[saicaca fuwari](https://github.com/saicaca/fuwari?tab=readme-ov-file)项目页面选择Use this template并Create a new repository，在Repository name下填入SiyuZhang-0426.github.io。

随后，使用git克隆创建的repo到本地：

```cmd
git clone https://github.com/SiyuZhang-0426/SiyuZhang-0426.github.io.git
```

接下来处理项目的依赖，在文件夹下右键选择Git Bash并运行：

```cmd
npm install -g pnpm
pnpm install
```

到此项目依赖处理完成，可通过运行以下命令预览项目：

```cmd
pnpm dev
```

随后访问`http://localhost:4321/`即可看到本地部署结果。

# Deployment

首先，在Github仓库中，点击Settings, Pages, 选择`Github Actions`项。

同时，在本地进入目录.github\workflows，新建deploy.yml配置文件，填写如下内容并保存：

```yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout your repository using git
        uses: actions/checkout@v4
      - name: Install, build, and upload your site
        uses: withastro/action@v3

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

最后，在根目录下启动Git Bash，运行：

```cmd
npm run build
git add .
git commit -m "Deploy to GitHub Pages"
git push origin main
```

到此Github Pages部署成功，访问`https://SiyuZhang-0426.github.io/`即可。