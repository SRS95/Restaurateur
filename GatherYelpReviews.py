## Gather Yelp reviews for a given restaurant
## author: Sam Schwager

import csv

def main():
	restaurant = raw_input("Restaurant name: ")
	print "Gathering Yelp reviews for", restaurant

	reviews = gatherReviews(restaurant)
	print "Successfully gathered Yelp reviews for", restaurant

	writeToCSV = raw_input("Would you like to write the Yelp reviews to a CSV file? (y/n)")
	fname = ""
	if writeToCSV == 'y':
		while(True):
			fname = raw_input("What would you like to name the file? (must end with .csv):")
			if (fname[len(fname) - 4:] == ".csv"):
				break

	#createCSV(reviews, fname)
	print "Successfully created", fname

def gatherReviews(restaurant):
	print "Need to call Yelp API or create scraper"

def createCSV(reviews, fname):
	with open(fname, 'wb') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    	for row in reviews:
    		spamwriter.writerow(row)

if __name__ == "__main__":
	main()
