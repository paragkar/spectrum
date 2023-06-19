from deta import Deta


DETA_KEY= st.secrets["deta_auth_tele_app"]

deta = Deta(DETA_KEY)

db = deta.Base("users_db")

def insert_user(username, name, password):

	#Returns the users on a successful user creation, othewise raises an error

	return db.put({"key" : username, "name": name, "password" : password})

insert_user("pparker", "Peter Parker", "abc123")



