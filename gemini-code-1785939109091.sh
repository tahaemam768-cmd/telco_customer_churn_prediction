# 1. Initialize Git in your project folder
git init

# 2. Add all files (app.py, requirements.txt, .pkl files) to staging
git add .

# 3. Commit the changes
git commit -m "Initial commit for churn prediction app"

# 4. Rename the default branch to main
git branch -M main

# 5. Link your local folder to your remote GitHub repository
# (Replace the URL below with your actual GitHub repo URL from Step 1)
git remote add origin https://github.com/YOUR-USERNAME/churn-prediction-app.git

# 6. Push your code to GitHub
git push -u origin main