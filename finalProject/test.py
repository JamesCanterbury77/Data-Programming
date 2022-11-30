from bs4 import BeautifulSoup

data = '<div class="div-col" style="column-width: 10em;">\n<ol><li>123456</li>\n<li>password</li>\n<li>12345678</li' \
       '>\n<li>qwerty</li>\n<li>123456789</li>\n<li>12345</li>\n<li>1234</li>\n<li>111111</li>\n<li>1234567</li>\n ' \
       '<li>dragon</li>\n<li>123123</li>\n<li>baseball</li>\n<li>abc123</li>\n<li>football</li>\n<li>monkey</li>\n ' \
       '<li>letmein</li>\n<li>696969</li>\n<li>shadow</li>\n<li>master</li>\n<li>666666</li>\n<li>qwertyuiop</li>\n' \
       '<li>123321</li>\n<li>mustang</li>\n<li>1234567890</li>\n<li>michael</li>\n<li>654321</li>\n<li>pussy</li>\n' \
       '<li>superman</li>\n<li>1qaz2wsx</li>\n<li>7777777</li>\n<li>fuckyou</li>\n<li>121212</li>\n<li>000000</li>\n' \
       '<li>qazwsx</li>\n<li>123qwe</li>\n<li>killer</li>\n<li>trustno1</li>\n<li>jordan</li>\n<li>jennifer</li>\n' \
       '<li>zxcvbnm</li>\n<li>asdfgh</li>\n<li>hunter</li>\n<li>buster</li>\n<li>soccer</li>\n<li>harley</li>\n<li' \
       '>batman</li>\n<li>andrew</li>\n<li>tigger</li>\n<li>sunshine</li>\n<li>iloveyou</li>\n<li>fuckme</li>\n<li' \
       '>2000</li>\n<li>charlie</li>\n<li>robert</li>\n<li>thomas</li>\n<li>hockey</li>\n<li>ranger</li>\n<li>daniel' \
       '</li>\n<li>starwars</li>\n<li>klaster</li>\n<li>112233</li>\n<li>george</li>\n<li>asshole</li>\n<li>computer' \
       '</li>\n<li>michelle</li>\n<li>jessica</li>\n<li>pepper</li>\n<li>1111</li>\n<li>zxcvbn</li>\n<li>555555</li' \
       '>\n<li>11111111</li>\n<li>131313</li>\n<li>freedom</li>\n<li>777777</li>\n<li>pass</li>\n<li>fuck</li>\n<li' \
       '>maggie</li>\n<li>159753</li>\n<li>aaaaaa</li>\n<li>ginger</li>\n<li>princess</li>\n<li>joshua</li>\n<li' \
       '>cheese</li>\n<li>amanda</li>\n<li>summer</li>\n<li>love</li>\n<li>ashley</li>\n<li>6969</li>\n<li>nicole' \
       '</li>\n<li>chelsea</li>\n<li>biteme</li>\n<li>matthew</li>\n<li>access</li>\n<li>yankees</li>\n<li>987654321' \
       '</li>\n<li>dallas</li>\n<li>austin</li>\n<li>thunder</li>\n<li>taylor</li>\n<li>matrix</li></ol>\n</div>' \
    # print(data[54:])
soup = BeautifulSoup(data, features='lxml')
text = str(soup.find_all('li', text=True))
print(text)
passwords = []

while len(text) >= 4:
    close = text.index('>')
    text = text[close + 1:]
    # print(str(close))
    opens = text.index('<')
    # print(str(opens))
    # print(text)

    passwords.append(text[0:opens])
    text = text[opens + 7:]
    # print(text)

print('Passwords: ' + str(passwords))
