# Simple Deployment with AWS Elastic Beanstalk

This guide provides instructions for a simple deployment of the Credit Risk Prediction API to AWS Elastic Beanstalk.

## Prerequisites

Before you begin, ensure you have the following:

- **AWS Account:** You'll need an AWS account to use Elastic Beanstalk.
- **AWS CLI:** The AWS Command Line Interface (CLI) must be installed and configured on your local machine. For instructions, see the [AWS CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).
- **Elastic Beanstalk CLI (EB CLI):** The EB CLI is a command-line interface for AWS Elastic Beanstalk that provides interactive commands that simplify creating, updating, and monitoring environments from a local repository. For installation instructions, see the [EB CLI documentation](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html).

## Deployment Steps

### 1. Initialize the Elastic Beanstalk Application

Navigate to your project's root directory and run the following command to initialize your Elastic Beanstalk application:

```bash
eb init -p "Docker" --region us-east-1 credit-risk-app
```

This command initializes a new Elastic Beanstalk application named `credit-risk-app` in the `us-east-1` region and configures it to use the Docker platform.

### 2. Create the Elastic Beanstalk Environment

Next, create a new environment for your application with the following command:

```bash
eb create credit-risk-env
```

This command creates a new environment named `credit-risk-env` and automatically deploys your application to it. The deployment process may take a few minutes.

### 3. Verify the Deployment

Once the deployment is complete, you can open your application in a web browser with the following command:

```bash
eb open
```

This command opens your application's URL in your default web browser. You should see the message: `{"message":"Welcome to the Credit Risk Prediction API!"}`.

### 4. Test the Prediction Endpoint

To test the prediction endpoint, you can use the `predict-test.py` script. First, set the `PREDICTION_URL` environment variable to your application's URL, which you can get from the output of the `eb status` command. Then, run the script:

```bash
export PREDICTION_URL=$(eb status --verbose | grep CNAME | awk '{print $2}')
python predict-test.py
```

### 5. Clean Up

To avoid incurring unnecessary charges, you can terminate your environment with the following command:

```bash
eb terminate credit-risk-env
```

This command terminates the environment and all associated resources.


## Non-interactive SSH key generation

If you encounter the interactive prompt "Enter file in which to save the key" when generating SSH keys (e.g., in CI or automation), use the helper script to generate a key pair without prompts:

Usage:
- ./scripts/generate_ssh_key.sh idk

This will create idk and idk.pub using ssh-keygen with a blank passphrase and a specified output file, avoiding interactive prompts. If the files already exist, the script is a no-op.


## Roadmap and Next Steps

For a concise guide on what to do now, including developer quickstart, local and Docker run instructions, CI/CD and deployment, model lifecycle, and a prioritized checklist, see the project roadmap: [ROADMAP.md](./ROADMAP.md)


## API Documentation (/docs)

- After starting the API (locally or in Docker), open http://localhost:9696/docs for interactive Swagger UI.
- The raw OpenAPI spec is available at http://localhost:9696/openapi.json.

## Pre-commit Hooks

To keep code quality consistent locally, install and enable pre-commit:

```bash
pip install pre-commit
pre-commit install
# Run against all files once
pre-commit run --all-files
```

These hooks run Black, isort, and Flake8 before each commit.


## Troubleshooting: Not seeing updates on GitHub?

If your changes aren’t appearing on GitHub, check these steps:

1) Commit your changes locally
- git status
- git add -A
- git commit -m "Your message"

2) Verify the current branch and push it
- git branch --show-current  # shows the active branch (e.g., main or master)
- git push -u origin HEAD    # pushes the current branch and sets upstream

3) Check your remote
- git remote -v
If none or incorrect, set it:
- git remote add origin git@github.com:<your-username>/<your-repo>.git  # SSH
  or
- git remote add origin https://github.com/<your-username>/<your-repo>.git  # HTTPS

4) Using SSH? Ensure your key is set up
- Generate a key (non-interactive): ./scripts/generate_ssh_key.sh idk
- Copy idk.pub content and add it at GitHub → Settings → SSH and GPG keys → New SSH key
- Test: ssh -T git@github.com (you should see a success message)

5) GitHub Actions not running?
- Workflow files must be in .github/workflows/*.yml (fixed in this repo)
- Push to the default branch (main or master); then check the Actions tab

6) Still not seeing updates?
- Confirm you’re looking at the same branch on GitHub as the one you pushed
- Refresh the page; check commit history and branch selector on GitHub
