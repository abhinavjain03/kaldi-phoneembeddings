mess=$1
whatToCommit=$2

git add $whatToCommit
git commit -m "$mess"
git push origin master