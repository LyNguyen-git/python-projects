
import random

def drawing_man(line_to_draw,letter_is_right):
    man =''
    if line_to_draw == 0:
        man = '\n: '+ ' '*5 +'d0_0b'
        man+= '\n: '+ ' '*5 +' /|\\'
        man+= '\n: '+ ' '*5 +' / \\'
    elif line_to_draw < 10:
        man = '\n: '+ ' '*5 +'d-_-b'
        man+= '\n: '+ ' '*5 +'  []'
        man+= '\n: '+ ' '*5 +'  ||'
        if letter_is_right:
            man = '\n: '+ ' '*5 +'d0_ob'
            man+= '\n: '+ ' '*5 +' /\\'
            man+= '\n: '+ ' '*5 +' ||'
    elif line_to_draw == 10:
        man = '\n|'+ ' '*5 +'d+__+b'
        man+= '\n:'+ ' '*5+'   ||'
        man+= '\n:'+ ' '*5+'  ||'
    return man
    
def man_in_frame(line_to_draw,guess_right):
    man_in_frame = ''
    line = ''
    if line_to_draw <= 4:
        line = '\n|'*line_to_draw
        man_in_frame = line + drawing_man(line_to_draw,guess_right)
    elif line_to_draw <= 7:
        line = '___'*(line_to_draw-4)+ '\n|'*4
        man_in_frame = line + drawing_man(line_to_draw,guess_right)
    elif line_to_draw > 7:
        i = line_to_draw - 7
        line = '___'*3 
        line += ('\n|'+' '*7+ '|')*i
        line += '\n|'*(4-i)

        if line_to_draw != 10:
            man_in_frame = line + drawing_man(line_to_draw,guess_right)
        else:
            line = '___'*3 
            line += ('\n|'+' '*7+ '|')*3
            man_in_frame = line + drawing_man(line_to_draw,guess_right)
            man_in_frame += '\n:'
    return man_in_frame

def hidden_word(guessed_true,word):
    hidden_w = ""
    for i in word:
        if i in guessed_true:
            hidden_w += i + " "
        else:
            hidden_w += "_ "
    return hidden_w      

def letter_in_word(letter,word):
    letter_status = True
    if letter.upper() not in word:
        letter_status = False
    return letter_status

def select_word():
    L = ['HOUSE','BOOK','COFFEE','MOON','CAT','TODAY','GAME','COMPUTER','PAPER','KITCHEN']
    n = random.randint(0,len(L)-1)
    return L[n]

def GameOn():
    word = select_word()
    guessed_true = []
    hidden_w = hidden_word(guessed_true,word)
    allready_guessed = []
    times = 10
    line_to_draw = 0 
    
    print(f'''\nGuess this word : {hidden_w}
    \nYour have {times} times to guess''')
    print(man_in_frame(line_to_draw,True))  
    
    while times >= 1:
        letter = input('Enter a letter: ')
        print('............')
        letter_is_right = letter_in_word(letter,word)
        if letter in allready_guessed:
            print(f'\nThe letter \'{letter.upper()}\' is ALLREADY guessed!')
            hidden_w = hidden_word(guessed_true,word)
        else:
            allready_guessed.append(letter)
            if letter_is_right:
                guessed_true.append(letter.upper())
                hidden_w = hidden_word(guessed_true,word)
                print('\nYou were RIGHT!')
            else:
                times -= 1  
                line_to_draw += 1
                hidden_w = hidden_word(guessed_true,word)
                print(f'\nThere is NO LETTER \'{letter.upper()}\' in this word!')  
        print(f'The word is now   :  {hidden_w}')
        hidden_w = hidden_w.replace(' ','').replace('_','')
        if hidden_w == word:
            print('\nYOU WIN!')
            print(man_in_frame(0,True))
            break
        print(f'Times to guess: {times} ')
        print(man_in_frame(line_to_draw,letter_is_right))
    if hidden_w != word:
        print('\nYOU LOSE!')
        print(f'The word is \'{word}\' ')  

def main():
    while True:
        print('''\n\n***********************\nHANGING MAN GAME ''')
        GameOn()
        print('''\n***********************''')
        command = input("Type a command ('Q' to quit, Enter to play again): ")
        if command.upper() == 'Q':
            break
        else:
            continue

if __name__== "__main__":
    main()