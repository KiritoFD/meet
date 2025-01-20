# 房间控制组件

## 功能说明
提供房间创建、加入、管理等功能的可复用组件。

## 组件结构
```html
<div class="room-controls">
    <!-- 房间创建 -->
    <div class="create-room">
        <button id="createRoomBtn">Create Room</button>
        <input type="text" id="roomIdInput" placeholder="Room ID">
    </div>
    
    <!-- 成员列表 -->
    <div class="member-list">
        <h3>Members</h3>
        <ul id="memberList"></ul>
    </div>
    
    <!-- 房间设置 -->
    <div class="room-settings">
        <label>
            <input type="checkbox" id="autoCloseCheck">
            Auto Close
        </label>
        <select id="maxMembersSelect">
            <option value="5">5 members</option>
            <option value="10">10 members</option>
        </select>
    </div>
</div>
```

## JavaScript API

### 房间管理
```javascript
class RoomControls {
    constructor(socket, options = {}) {
        this.socket = socket;
        this.options = {
            maxMembers: 10,
            autoClose: true,
            ...options
        };
        this.currentRoom = null;
        this.members = new Map();
        
        this.initEventListeners();
    }
    
    // 创建房间
    async createRoom() {
        const roomId = this.generateRoomId();
        const success = await this.socket.emit('create_room', {
            room_id: roomId,
            settings: this.options
        });
        if (success) {
            this.currentRoom = roomId;
            this.updateUI();
        }
    }
    
    // 更新成员列表
    updateMemberList(members) {
        const list = document.getElementById('memberList');
        list.innerHTML = '';
        members.forEach(member => {
            const li = document.createElement('li');
            li.textContent = member.name;
            li.dataset.id = member.id;
            list.appendChild(li);
        });
    }
}
```

### 事件处理
```javascript
// 成员加入事件
socket.on('member_join', (data) => {
    const { member_id, member_name } = data;
    this.members.set(member_id, {
        id: member_id,
        name: member_name,
        joinTime: Date.now()
    });
    this.updateMemberList();
});

// 成员离开事件
socket.on('member_leave', (data) => {
    const { member_id } = data;
    this.members.delete(member_id);
    this.updateMemberList();
});
```

## 样式定义
```css
.room-controls {
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.member-list {
    max-height: 200px;
    overflow-y: auto;
}

.member-list li {
    padding: 5px 10px;
    border-bottom: 1px solid #eee;
}

.room-settings {
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid #eee;
}
```

## 使用示例
```javascript
// 初始化组件
const roomControls = new RoomControls(socket, {
    maxMembers: 5,
    autoClose: true,
    onMemberJoin: (member) => {
        console.log(`${member.name} joined`);
    },
    onMemberLeave: (member) => {
        console.log(`${member.name} left`);
    }
});

// 创建房间
document.getElementById('createRoomBtn').onclick = () => {
    roomControls.createRoom();
}; 