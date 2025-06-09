css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Flocalo.com%2Fmarketing-dictionary%2Fwhat-is-bot&psig=AOvVaw0GMCMLZlplSegjuy8a1aGR&ust=1741862450971000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJDqz7SthIwDFQAAAAAdAAAAABAE">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fpixabay.com%2Fvectors%2Fperson-individually-alone-icon-1824144%2F&psig=AOvVaw0ezQyyxtvNxNZpc_-Kbgly&ust=1741862598730000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOCm3f6thIwDFQAAAAAdAAAAABAE">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''