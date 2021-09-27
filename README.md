# Yelp Map
Created a visualization of restaurant ratings using machine learning
techniques such as k-means algorithm, simple least-squares linear
regression, and the Yelp academic dataset.

## Run Locally
Clone the project
```bash
git clone https://github.com/xingvoong/yelp_map
```
Go to the project directory
```bash
cd yelp_map
```
Run:

- `-u` : to select a user from the users directory
```bash
python3 recommend.py -u one_cluster
```
- `-k` : to get more fine-grained groupings by increasing the number of clusters with the -k option:
```bash
python3 recommend.py -k 2
python3 recommend.py -u likes_everything -k 3
```
	

- to predict what rating a user would give a restaurant:
```bash
python3 recommend.py -u likes_southside -k 5 -p
```
	
	
- `-q`: to filter based on a category.

- to visualize all sandwich restaurants and their predicted ratings for the user who likes_expensive restaurants:
```bash
python3 recommend.py -u likes_expensive -k 2 -p -q Sandwiches
```
## Acknowledgements
- [UC Berkeley CS61A](https://cs61a.org/)
