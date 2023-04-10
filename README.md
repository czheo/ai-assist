A command-line AI assist that can do math, search, run Python and shell command.
**Under experiment. Use at your own risks.**

# Setup
You will need put the following environment variables in a `.env` file in your home directory.

```
# used for calling OpenAI's API
OPENAI_API_KEY=...

# used for calling Google Search API
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
```

You can generate/get keys from links below:

- `OPENAI_API_KEY` from https://platform.openai.com/account/api-keys
- `GOOGLE_API_KEY` from https://console.cloud.google.com/apis/credentials
- `GOOGLE_CSE_ID` from https://programmablesearchengine.google.com/controlpanel/all

# Usage

```shell
$ ai

Welcome to the AI.
I do math, search, run python/bash and more.
Type 'exit' to quit.
[USER]<< What's the largest prime number less than 1000?
...
[AI]>> 997
```
