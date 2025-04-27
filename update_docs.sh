#!/bin/bash

cd docs || exit

echo "🔄 Rebuilding Sphinx HTML..."
make html

echo "📦 Copying build to deploy folder..."
cp -r _build/html/. .

cd ..

echo "✅ Ready to commit. Now run:"
echo "   git add docs/"
echo "   git commit -m 'Update docs'"
echo "   git push origin main"
