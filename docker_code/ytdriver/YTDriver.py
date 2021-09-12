from selenium.webdriver import Chrome, ChromeOptions, FirefoxOptions, Firefox
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from time import sleep
import re
import requests
from pyvirtualdisplay import Display

class Video:
    def __init__(self, elem, url, title_elem=None):
        self.elem = elem
        self.url = url
        self.videoId = re.search(r'\?v=(.*)?$', url).group(1)
        self.title_elem = title_elem

    def get_metadata(self):
        self.title = "TODO"
        self.comments = ["TODO"]
        #self.__get_metadata()
        self.__get_slant()
        #self.__get_comments()
        
    def get_mean_channel_slant(self):
        self.__get_channel_slant()
        
    def __get_metadata(self):
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getMetadata/%s' % self.videoId)
        #r = requests.get('http://localhost:5000/getMetadata/%s' % self.videoId)
        if r.status_code == 200:
            self.metadata = r.json()
    
    def __get_slant(self):
        #print(self.videoId)
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getSlant/%s' % self.videoId)
        #r = requests.get('http://localhost:5000/getSlant/%s' % self.videoId)
        if r.status_code == 200:
            js = r.json()
            self.score = js.get('slant', None)
            self.conservative_landmarks = js.get('conservative_landmark_follows', None)
            self.liberal_landmarks = js.get('liberal_landmark_follows', None)
            
            if self.conservative_landmarks + self.liberal_landmarks < 12:
                self.score = None
                self.conservative_landmarks = None
                self.liberal_landmarks = None
                
        else:
            self.score = None
            self.conservative_landmarks = None
            self.liberal_landmarks = None
            
    def __get_comments(self):
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getComments/%s' % self.videoId)
        #r = requests.get('http://localhost:5000/getComments/%s' % self.videoId)
        if r.status_code == 200:
            js = r.json()
            self.comments = js.get('comments', [])
        else:
            self.comments = []
            
    def __get_channel_slant(self):
        #r = requests.get('http://localhost:5000/getChannelSlant/%s' % self.videoId)
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getChannelSlant/%s' % self.videoId)
        if r.status_code == 200:
            self.score = r.json().get('mean_slant', None)


class YTDriver:

    def __init__(self, profile_dir=None, use_virtual_display=False, headless=False, verbose=False):
        options = FirefoxOptions()
        options.add_argument('--window-size=1920,1080')
        if use_virtual_display:
            Display(size=(1920,1080)).start()
        if profile_dir is not None:
            pass
        if headless:
            options.add_argument('--headless')

        self.verbose = verbose
        self.driver = Firefox(options=options)

    def close(self):
        self.driver.close()

    def get_homepage(self, iteration=None):
        self.driver.get('https://youtube.com')
        homepage = []
        sleep(1)
        
        if iteration is not None:
            for i in range(8):
                self.driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
                self.driver.save_screenshot('./HOMEPAGES/' + str(iteration) + '-' + str(i) + '.png')
                sleep(0.5)
            
        # scroll page to load more results
        for _ in range(8): #15
            self.driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
            sleep(0.6) #0.2

        # collect video-like tags from homepage
        videos = self.driver.find_elements_by_xpath('//div[@id="primary"]/ytd-rich-grid-renderer/div[6]/ytd-rich-item-renderer')

        # identify actual videos from tags
        for video in videos:
            a = video.find_elements_by_tag_name('a')[0]
            
            #title_elem = video.find_elements_by_tag_name('a')[2]
            title_elem = video.find_element(By.ID, "video-title")
            
            href = a.get_attribute('href')
            if href is not None and href.startswith('https://www.youtube.com/watch?'):
                #########################
                href = href.split('&')[0]
                #########################
                homepage.append(Video(a, href, title_elem))
        return homepage
    
    def play_video(self, video, duration=1):
        # this function returns when the video starts playing
        try:
            self.__click_video(video)
            self.__click_play_button()
            self.__handle_ads()
            self.__clear_prompts()
            sleep(duration)
        except Exception as e:
            self.__log(e)

    def get_recommendations(self, topn=10):
        # fetch recommendation elements
        elems = self.driver.find_elements_by_tag_name('ytd-compact-video-renderer')
        
        # recommended videos array
        return [Video(elem, elem.find_elements_by_tag_name('a')[0].get_attribute('href')) for elem in elems[:topn]]
    
    
    def exposed_ad_handler(self):
        self.__handle_ads()
        

    ## helper methods
    def __log(self, message):
        if self.verbose:
            print(message)

    def __click_video(self, video):
        if type(video) == Video:
            try:
                # try to click the element using selenium
                self.__log("Clicking element via Selenium...")
                video.elem.click()
                return
            except Exception as e:
                print(e)
                try:
                    # try to click the element using javascript
                    self.__log("Failed. Clicking via Javascript...")
                    self.driver.execute_script('arugments[0].click()', video.elem)
                except:
                    # js click failed, just open the video url
                    self.__log("Failed. Loading video URL...")
                    self.driver.get(video.url)
        elif type(video) == str:
            self.driver.get(video)

    def __click_play_button(self):
        try:
            playBtn = self.driver.find_elements_by_class_name('ytp-play-button')
            if 'Play' in playBtn[0].get_attribute('title'):
                playBtn[0].click()
        except:
            pass

    def __handle_ads(self):
        # handle multiple ads
        while True:
            sleep(1)

            # check if ad is being shown
            preview = self.driver.find_elements_by_class_name('ytp-ad-preview-container')
            if len(preview) == 0:
                self.__log('Ad not detected')
                # ad is not shown, return
                return

            self.__log('Ad detected')
            
            sleep(1)
            preview = preview[0]
            # an ad is being shown
            # grab preview text to determine ad type
            text = preview.text.replace('\n', ' ')
            wait = 0
            if 'after ad' in text:
                # unskippable ad, grab ad length
                length = self.driver.find_elements_by_class_name('ytp-ad-duration-remaining')[0].text
                wait = time2seconds(length)
                self.__log('Unskippable ad. Waiting %d seconds...' % wait)
            elif 'begin in' in text or 'end in' in text:
                # short ad
                wait = int(text.split()[-1])
                self.__log('Short ad. Waiting for %d seconds...' % wait)
            else:
                # skippable ad, grab time before skippable
                wait = int(text)
                self.__log('Skippable ad. Skipping after %d seconds...' % wait)

            # wait for ad to finish
            sleep(wait)

            # click skip button if available
            skip = self.driver.find_elements_by_class_name('ytp-ad-skip-button-container')
            if len(skip) > 0:
                skip[0].click()

    def __clear_prompts(self):
        try:
            sleep(1)
            self.driver.find_element_by_xpath('/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/yt-tooltip-renderer/div[2]/div[1]/yt-button-renderer/a/tp-yt-paper-button/yt-formatted-string').click()
        except:
            pass

def time2seconds(s):
    s = s.split(':')
    s.reverse()
    wait = 0
    factor = 1
    for t in s:
        wait += int(t) * factor
        factor *= 60
    return wait
