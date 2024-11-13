from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
import pandas as pd
from time import sleep
import random
# Cấu hình tùy chọn cho Edge
options = Options()
options.add_argument("--start-maximized")  # Mở ở chế độ toàn màn hình
options.add_argument("--inprivate")  # Chế độ ẩn danh (InPrivate mode)
driver = webdriver.Edge(options=options)
driver.get("https://www.facebook.com/groups/ued.confessions?locale=vi_VN")
# username = driver.find_element(By.CSS_SELECTOR, '.x1egiwwb.x4l50q0 .xod5an3 input')
# password = driver.find_elements(By.CSS_SELECTOR, '.x1egiwwb.x4l50q0 .x1c436fg input')[0]
# login = driver.find_elements(By.CSS_SELECTOR, '.x1egiwwb.x4l50q0 .x1c436fg')[1]
# username.send_keys('khoalolriot@gmail.com')
# password.send_keys('Akhoa@6204')
# login.click()
def scroll(driver):
    driver.execute_script("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});")
    sleep(1)
def scroll_to_post(driver, post):
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", post)
    sleep(1)
def scroll_to_top_post(driver, post): 
    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'top'});", post)
    sleep(1)
def countScroll(driver, scroll):
    count = 0
    while count < 31: 
        scroll(driver)
        count += 1
df = pd.DataFrame(columns=['index', 'linkPoster', 'name', 'role', 'time', 'content', 'content_img', 'linkPost', 'status'])
df_video = pd.DataFrame(columns=['index', 'linkPoster', 'name', 'time', 'content', 'linkVideo', 'typeVideo', 'status'])

index  = 0
post_list = driver.find_elements(By.CSS_SELECTOR, '[role="feed"] .x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z')
while True: 
    if index >= len(post_list):  
        print(len(post_list))
        countScroll(driver, scroll)
        post_list_new = driver.find_elements(By.CSS_SELECTOR, '[role="feed"] .x1yztbdb.x1n2onr6.xh8yej3.x1ja2u2z')
        if len(post_list_new) == len(post_list):
            print('Thu thập được:',len(post_list),'bài đăng')
            break 
        else:
            post_list = post_list_new 
            print(len(post_list))  
            continue 
        
    try:
        main  = post_list[index].find_element(By.XPATH,'./div/div/div/div/div/div/div/div/div/div[13]/div/div')
        print('post normal')
        scroll_to_top_post(driver, post_list[index])
        print(f"Đã scroll tới bài đăng số {index + 1}:")
        
        info_box  = main.find_element(By.XPATH,'./div[2]')
        content_box = main.find_element(By.XPATH,'./div[3]')
        react_cmt_box = main.find_element(By.XPATH,'./div[4]')
        
        user_link = info_box.find_element(By.XPATH, './div/div[2]/div/div[1]').find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
        print(user_link)
        user_name = info_box.find_element(By.XPATH, './div/div[2]/div/div[1]').find_element(By.CSS_SELECTOR, '.xt0psk2').text  
        print(user_name)      
        time_post = info_box.find_element(By.XPATH, './div/div[2]/div/div[2]')
        print(time_post)
        scroll_to_post(driver, time_post)
        linkPost = time_post.find_element(By.CSS_SELECTOR, 'a')
        driver.execute_script("""
            var link = arguments[0];
            link.removeAttribute('target');
            link.addEventListener('click', function(event) {
                event.preventDefault();
                event.stopPropagation();
            });
        """, linkPost)
        
        try:
            linkPost.click()
        except: 
            scroll_to_top_post(driver, time_post)
            linkPost.click()
            
        linkPost = linkPost.get_attribute('href')
        print(linkPost)
        try: 
            role = time_post.find_element(By.CSS_SELECTOR,'.x3nfvp2.x1kgmq87').text
        except NoSuchElementException:
            role = None
        print(role)
        try: 
            time_link = time_post.find_element(By.CSS_SELECTOR, 'use').get_attribute('xlink:href')
            time_post = driver.find_element(By.CSS_SELECTOR, f'text{time_link}').get_attribute('textContent')
        except NoSuchElementException:  
            time_post = time_post.text.replace('·','').strip()
        print(time_post)    
        try: 
            main_content = content_box.find_element(By.XPATH, './div[1]')
            try: 
                more_content = main_content.find_element(By.CSS_SELECTOR, 'div[role="button"]').click()
            except NoSuchElementException: 
                pass
            main_content = content_box.find_element(By.XPATH, './div[1]').text
        except NoSuchElementException:
            main_content = None
        print(main_content)    
        content_img_post = {}        
        i = 0
        try:     
            content_img = content_box.find_element(By.XPATH, './div[2]')
            content_img_list = content_img.find_elements(By.CSS_SELECTOR, 'img')
            for img in content_img_list:
                content_img_post[i] = img.get_attribute('src')
        except NoSuchElementException: 
            pass
        print(content_img_post)   
        post_data = {
            'index': index + 1,
            'linkPoster': user_link if user_name != 'Người tham gia ẩn danh' else 'No link',
            'name': user_name,
            'role': role if role else 'Thành viên',
            'time': time_post,
            'content': main_content if main_content else 'No content',
            'content_img': content_img_post if content_img_post else 'No picture',
            'linkPost': linkPost if linkPost else 'error',
            'status': 'Success'
        }
        index += 1
        print(post_data)
    except IndexError: # video
        print('post video')
        href = post_list[index].find_element(By.CSS_SELECTOR, 'a').get_attribute('href')
        info = post_list[index].find_element(By.CSS_SELECTOR,'.x14l7nz5 span span span').get_attribute('textContent').split('·')
        typeVideo = info[0].replace('\xa0', '')
        linkPoster = post_list[index].find_element(By.CSS_SELECTOR,'.x1swvt13.x1y1aw1k a').get_attribute('href')
        name = info[1].replace('\xa0', '')
        timeVideo = info[2].replace('\xa0', '')
        content_post = post_list[index].find_element(By.CSS_SELECTOR,'.xjkvuk6.x1rz3hdg')
        try: 
            more_content = content_post.find_element(By.CSS_SELECTOR, 'div[role="button"]').click()
        except NoSuchElementException: 
            pass 
        content_post = post_list[index].find_element(By.CSS_SELECTOR,'.xjkvuk6.x1rz3hdg').get_attribute('textContent')

        post_data = {
            'index': index + 1,
            'linkPoster': linkPoster,
            'name': name,
            'time': timeVideo,
            'content': content_post if content_post else 'No content',
            'linkVideo': href,
            'typeVideo': typeVideo,
            'status': 'Success'
        }
        print(f'crawl bài đăng {index + 1} thành công')
        df_video.loc[len(df_video)] = post_data
        index += 1
    except Exception: 
        try:
            close = driver.find_element('xpath','/html/body/div[7]/div[1]/div/div[2]/div/div/div/div/div/div/div[3]/div/div/div/div/div/div/div/div[1]/div/span/span').click()
        except Exception: pass
    sleep(random.randrange(1))
    if index == 30: break 

# Xuất kết quả ra file CSV
df.to_csv('D:/post_csv', index=False, encoding='utf-8')
