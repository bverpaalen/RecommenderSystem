beginBold = "\033[1m"
endBold = "\033[0;0m"

def printHeader(title):
    print("")
    print("#######################################")
    print(beginBold + title + endBold)
    print("#######################################")

def printFooter(text):
    print("_______________________________________")
    print(text)
    print("_______________________________________")