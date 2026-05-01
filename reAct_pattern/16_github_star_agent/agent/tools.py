import requests
from langchain_core.tools import tool
from core.config import GITHUB_TOKEN

@tool
def get_github_stars(repo_name: str) -> str:
    """
    Fetches the number of stars for a given GitHub repository using the GitHub API.

    Args:
        repo_name (str): The full name of the repository in the format "owner/repo (e.g., "octocat/Hello-World").
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Use token if available to increase rate limits, but it works without one for public repos
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{repo_name}"

    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print(f"GitHub API response for {repo_name}: {data}")  # Debugging log
            stars = data.get("stargazers_count", "N/A")
            return f"The repository '{repo_name}' has {stars} stars."
        elif response.status_code == 404:
            return f"Repository '{repo_name}' not found."
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"An error occurred while fetching data from GitHub: {str(e)}"