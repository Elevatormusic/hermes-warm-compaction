# Security policy

## Supported versions

Only the latest release gets security fixes.

## Report a vulnerability

Do not open a public issue for a security problem.

Use GitHub private vulnerability reporting: go to the **Security** tab of this repository and click **Report a vulnerability**. Give the steps to reproduce, the effect, and the plugin and Hermes Agent versions. Do not include real API keys or private conversations.

You get a first reply within 7 days. When the fix is ready, we publish a security advisory and credit you, unless you ask us not to.

## Scope

The plugin sends the warm request with the credentials and headers of the Hermes main-model route, and it keeps the last request of each session in memory. Problems in these areas are in scope, for example:

- a request that goes to a different route, key, or session than the captured one;
- conversation text or credentials in logs, evidence files, or error messages;
- a summary that replaces history it does not describe.

Problems in Hermes Agent itself go to the [Hermes Agent repository](https://github.com/NousResearch/hermes-agent/security).
