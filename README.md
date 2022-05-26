# delegate-contributions-dp02
This repository is for DP0 delegates to contribute and share DP0.2-related materials (e.g., code, analysis tools, tutorials).

Tutorials could be Jupyter Notebooks, or markdown-formatted demonstrations of a DP0-related analysis with command-line tasks or the Portal Aspect.

Please contribute only code that you have verified to run.

## How to contribute to this repository.

### Workflow

1. Make a branch for your work, and name it u/(your github username).
2. Add and edit content in your branch until you're ready to share it with everyone.
3. Start a new Pull Request (PR) from your branch to main.
4. Request a review from another delegate or Rubin staff (safety feature; see below).
6. Merge your branch into ``main``.

Do not commit changes directly to ``main``.

### Reviews

For this repository, all PRs to merge to the ``main`` branch must be reviewed.
Reviewers should confirm that the PR is adding or editing only the user's own files in their own folder.
Reviewers should not evaluate the additions or changes: this is not a code review nor a peer review.
These reviews are just a saftey feature to avoid, e.g., this README.md being accidentally deleted (which has happened).
If the PR is deleting a file the reviewer should confirm that the user intended this deletion.
Reviewers can provide comments and approve the PR, but should not edit files, merge, or close the PR.

Any delegate who is unsure about how to request a review for a PR, or needs a reviewer, can ask Melissa Graham.

### Directories

All materials should be organized into directories.
An ``example`` directory is provided as an illustration.
Anyone may make a directory for a specific science topic, a specific tool or type of analysis, etc.
Directory creators should create a README.md file within the directory that identifies them as the directory maintainer and describes the contents of the directory.
Contact the directory creator if you want to contribute content to their directory.
Direct messages between delegates in Community.lsst.org is an appropriate mode of contact.

### Best Practices

Please always:
- document your code for non-experts
- provide links to background information
- clear notebook outputs before committing changes

All tutorials should have a header containing:
 - the author's name
 - the date the tutorial was last tested
 - the goals of the notebook

See the example directory for guidance on formatting notebooks and markdown files.

## Need to learn how to use GitHub?

Git is already installed in the Notebook Aspect of the Rubin Science Platform.

The best place to start in the extensive GitHub documentation is with the <a href="https://docs.github.com/en/get-started/quickstart/set-up-git">quickstart setup guide</a>.
There is also a <a href="https://training.github.com/downloads/github-git-cheat-sheet/">GitHub Cheat Sheet</a> of commonly used commands, and a <a href="https://docs.github.com/en/get-started/quickstart/github-glossary">GitHub Glossary</a>. 

<a href="https://www.youtube.com/watch?v=HVsySz-h9r4">Git Tutorial for Beginners: Command-Line Fundamentals</a> (a YouTube tutorial that includes git command line basics, but if you are not installing Git, you might want to skip a section of it describing the installation). 

<a href="https://medium.com/@christo8989/what-college-students-should-learn-about-git-6bbf6eaac39c">What college students should learn about Git</a> (a medium.com article that includes fundamental git concepts and basic git commands).

<a href="https://github.com/drphilmarshall/GettingStarted">Phil Marshall's notes on "Getting Started with git and GitHub"</a>.

<a href="https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent">Generating a new SSH key and adding it to the ssh-agent</a>, a guide to setting up git credentials so that you can push changes back to your remote repositories.

**See also the <a href="https://github.com/rubin-dp0/delegate-contributions-dp01/blob/main/CheatSheet.txt">GitHub Cheat Sheet</a> provided in this repository.** This cheat sheet was developed by Douglas Tucker for the <a href="https://github.com/LSSTScienceCollaborations/StackClub">LSST Science Collaborations Stack Club</a> and altered to be appropriate for Data Preview 0 by Greg Madejski. 

### Too many notifications?
<a href="https://docs.github.com/en/account-and-profile/managing-subscriptions-and-notifications-on-github/managing-subscriptions-for-activity-on-github/managing-your-subscriptions#choosing-how-to-unsubscribe">How to unsubscribe / unwatch GitHub repositories.</a>
