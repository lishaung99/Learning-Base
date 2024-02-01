ssh-keygen -t rsa -C “自己的邮箱地址”

ssh-keygen -t rsa -C '自己的邮箱地址' -f ~/.ssh/gitlab_rsa

git config --global user.name “你的用户名”
git config --global user.email “自己的邮箱地址”

git config --global --list

git remote -v

 git clone <你的项目地址>

git config user.name “OS开拓者”
git config user.email “36325396@qq.com”

git config  credential.helper store

git中在本地目录下关联远程库
git remote add origin <你的项目地址>

git init

git add .

git commit -m '提交'
git fetch origin 可以运行git fetch origin来同步远程服务器上的数据到本地
git push (origin master)
在执行push操作前，一定要有pull的操作，不管是那一分支都应该有pull的操作，所以，我在merge的时候，会将master pull下来，然后在执行git push --force 地址的操作
git push --force <你的项目地址>

//切换分支
git checkout (master develop release)
//合并分支
git merge develop master

//缓存所有文件（除了.gitignore中声名排除的）
git add -A

//提交跟踪过的文件（Commit the changes）
git commit -am "commit message"

//删除master分支（Delete the branch）
git branch -D master

//重命名当前分支为master
git branch -m master

//提交到远程master分支 （Finally, force update your repository）
git push -f origin master

git remote prune origin删除本地有但在远程库已经不存在的分支
先调用git remote show origin

该命令能够获取远端分支信息，你可以看到和本地和远端不同步的地方：

过时的就是和本地不同步的分支，本地已过时的表示你需要移除这个分支了。

这个时候你需要调用

git remote prune origin
同步远程的分支到本地，这样远程已经被删除的分支，本地就不会再看见了。
这样就删除本地有但在远程库已经不存在的分支

很多远程分支可能已经删除掉了，但是在vscode中还会继续缓存这些分支，下面的命令是批量删除本地这类分支：
git fetch --prune



1、git branch
显示所有的本地分支。

2、git branch -r
显示所有的远程分支

3、git branch -a
显示所有的分支，远程分支和本地分支

4、git branch newbranch
创建新的分支，但是不切换过去

5、git checkout main
分支切换，切换到分支branchname 上去

6、git checkout -b branchname
创建新的分支branchname，并且切换过去

7、git branch -D branchname 删除本地分支 （D 强制）

8、git branch -d -r branchname 删除远端分支

删除远程分支：git push origin --delete 分支名（remotes/origin/分支名）



git reset --hard xxxx(commit id) // 回退到指定版本
//commit id 可以通过git log查看

git reset --hard HEAD^ //回退到上个版本