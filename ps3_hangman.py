# Hangman game

import random
import string

WORDLIST_FILENAME = "words.txt"

def loadWords():
    print("Loading word list from file...")
    inFile = open(WORDLIST_FILENAME, 'r')
    line = inFile.readline()
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist

def chooseWord(wordlist):
    return random.choice(wordlist)
    
def isWordGuessed(secretWord, lettersGuessed):
    for ch in secretWord:
        if ch in lettersGuessed:
            continue
        else:
            return False
    return True

def getGuessedWord(secretWord, lettersGuessed):
    result =''
    for ch in secretWord:
        if ch in lettersGuessed:
            result += ch
        else:
            result += '_'
        result += ' '
    return result


def getAvailableLetters(lettersGuessed):
    return ''.join([ch for ch in string.ascii_lowercase if ch not in lettersGuessed])
    

def hangman(secretWord):
    
    print('Welcome to Hangman!')
    
    print('The Secret Word contains ',len(secretWord), ' letters.')
    
    lettersGuessed = []
    attemptsLeft = 8  # Liczba prób

    while attemptsLeft > 0:
        print('\n' + '-' * 10)
        print('You have', attemptsLeft, ' guesses left.')
        print('Available letters:', getAvailableLetters(lettersGuessed))
        
        guess = input('Please guess a letter: ').lower()
        
        if len(guess) != 1 or guess not in string.ascii_lowercase:
            print("Invalid input. Please enter a single letter.")
            continue
        
        if guess in lettersGuessed:
            print("You've already guessed that letter!")
        else:
            lettersGuessed.append(guess)

        if guess in secretWord:
            print("\nGood guess!")
            if isWordGuessed(secretWord, lettersGuessed):
                print('\nCongratulations! You won! The word was ', secretWord)
                return
            else:
                print("Your word now looks like this: ", getGuessedWord(secretWord, lettersGuessed))
                
        else:
            print("\nOops! That letter is not in the word.")
            print("Your word now looks like this: ", getGuessedWord(secretWord, lettersGuessed))

            
        attemptsLeft -= 1

    print('\nSorry, you lost! The secret word was ', secretWord)



wordlist = loadWords()
secretWord = chooseWord(wordlist).lower()
hangman(secretWord)


        
