- Created a visualization of restaurant ratings using machine learning
techniques such as k-means algorithm, simple least-squares linear
regression, and the Yelp academic dataset.

-u: to select a user from the users directory
python3 recommend.py -u one_cluster

- to get more fine-grained groupings by increasing the number of clusters 
	with the -k option:
	python3 recommend.py -k 2
	python3 recommend.py -u likes_everything -k 3

- to predict what rating a user would give a restaurant:
	python3 recommend.py -u likes_southside -k 5 -p
	
-q: to filter based on a category.

- to visualize all sandwich restaurants and 
	their predicted ratings for the user who likes_expensive restaurants:
	python3 recommend.py -u likes_expensive -k 2 -p -q Sandwiches
