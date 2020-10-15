;;                #####
;;                #   #
;; #######  ####### # #
;; #     #  #  #      #
;; #  @  ####  #     ####
;; #  #    ....## ####  #
;; #    ##### ## $$ $ $ #
;; ######   #           #
;;          #  ##########
;;          ####

(define (problem p99-microban-sequential)
  (:domain sokoban-sequential)
  (:objects
    dir-down - direction
    dir-left - direction
    dir-right - direction
    dir-up - direction
    player-01 - player
    pos-01-01 - location
    pos-01-02 - location
    pos-01-03 - location
    pos-01-04 - location
    pos-01-05 - location
    pos-01-06 - location
    pos-01-07 - location
    pos-01-08 - location
    pos-01-09 - location
    pos-01-10 - location
    pos-02-01 - location
    pos-02-02 - location
    pos-02-03 - location
    pos-02-04 - location
    pos-02-05 - location
    pos-02-06 - location
    pos-02-07 - location
    pos-02-08 - location
    pos-02-09 - location
    pos-02-10 - location
    pos-03-01 - location
    pos-03-02 - location
    pos-03-03 - location
    pos-03-04 - location
    pos-03-05 - location
    pos-03-06 - location
    pos-03-07 - location
    pos-03-08 - location
    pos-03-09 - location
    pos-03-10 - location
    pos-04-01 - location
    pos-04-02 - location
    pos-04-03 - location
    pos-04-04 - location
    pos-04-05 - location
    pos-04-06 - location
    pos-04-07 - location
    pos-04-08 - location
    pos-04-09 - location
    pos-04-10 - location
    pos-05-01 - location
    pos-05-02 - location
    pos-05-03 - location
    pos-05-04 - location
    pos-05-05 - location
    pos-05-06 - location
    pos-05-07 - location
    pos-05-08 - location
    pos-05-09 - location
    pos-05-10 - location
    pos-06-01 - location
    pos-06-02 - location
    pos-06-03 - location
    pos-06-04 - location
    pos-06-05 - location
    pos-06-06 - location
    pos-06-07 - location
    pos-06-08 - location
    pos-06-09 - location
    pos-06-10 - location
    pos-07-01 - location
    pos-07-02 - location
    pos-07-03 - location
    pos-07-04 - location
    pos-07-05 - location
    pos-07-06 - location
    pos-07-07 - location
    pos-07-08 - location
    pos-07-09 - location
    pos-07-10 - location
    pos-08-01 - location
    pos-08-02 - location
    pos-08-03 - location
    pos-08-04 - location
    pos-08-05 - location
    pos-08-06 - location
    pos-08-07 - location
    pos-08-08 - location
    pos-08-09 - location
    pos-08-10 - location
    pos-09-01 - location
    pos-09-02 - location
    pos-09-03 - location
    pos-09-04 - location
    pos-09-05 - location
    pos-09-06 - location
    pos-09-07 - location
    pos-09-08 - location
    pos-09-09 - location
    pos-09-10 - location
    pos-10-01 - location
    pos-10-02 - location
    pos-10-03 - location
    pos-10-04 - location
    pos-10-05 - location
    pos-10-06 - location
    pos-10-07 - location
    pos-10-08 - location
    pos-10-09 - location
    pos-10-10 - location
    pos-11-01 - location
    pos-11-02 - location
    pos-11-03 - location
    pos-11-04 - location
    pos-11-05 - location
    pos-11-06 - location
    pos-11-07 - location
    pos-11-08 - location
    pos-11-09 - location
    pos-11-10 - location
    pos-12-01 - location
    pos-12-02 - location
    pos-12-03 - location
    pos-12-04 - location
    pos-12-05 - location
    pos-12-06 - location
    pos-12-07 - location
    pos-12-08 - location
    pos-12-09 - location
    pos-12-10 - location
    pos-13-01 - location
    pos-13-02 - location
    pos-13-03 - location
    pos-13-04 - location
    pos-13-05 - location
    pos-13-06 - location
    pos-13-07 - location
    pos-13-08 - location
    pos-13-09 - location
    pos-13-10 - location
    pos-14-01 - location
    pos-14-02 - location
    pos-14-03 - location
    pos-14-04 - location
    pos-14-05 - location
    pos-14-06 - location
    pos-14-07 - location
    pos-14-08 - location
    pos-14-09 - location
    pos-14-10 - location
    pos-15-01 - location
    pos-15-02 - location
    pos-15-03 - location
    pos-15-04 - location
    pos-15-05 - location
    pos-15-06 - location
    pos-15-07 - location
    pos-15-08 - location
    pos-15-09 - location
    pos-15-10 - location
    pos-16-01 - location
    pos-16-02 - location
    pos-16-03 - location
    pos-16-04 - location
    pos-16-05 - location
    pos-16-06 - location
    pos-16-07 - location
    pos-16-08 - location
    pos-16-09 - location
    pos-16-10 - location
    pos-17-01 - location
    pos-17-02 - location
    pos-17-03 - location
    pos-17-04 - location
    pos-17-05 - location
    pos-17-06 - location
    pos-17-07 - location
    pos-17-08 - location
    pos-17-09 - location
    pos-17-10 - location
    pos-18-01 - location
    pos-18-02 - location
    pos-18-03 - location
    pos-18-04 - location
    pos-18-05 - location
    pos-18-06 - location
    pos-18-07 - location
    pos-18-08 - location
    pos-18-09 - location
    pos-18-10 - location
    pos-19-01 - location
    pos-19-02 - location
    pos-19-03 - location
    pos-19-04 - location
    pos-19-05 - location
    pos-19-06 - location
    pos-19-07 - location
    pos-19-08 - location
    pos-19-09 - location
    pos-19-10 - location
    pos-20-01 - location
    pos-20-02 - location
    pos-20-03 - location
    pos-20-04 - location
    pos-20-05 - location
    pos-20-06 - location
    pos-20-07 - location
    pos-20-08 - location
    pos-20-09 - location
    pos-20-10 - location
    pos-21-01 - location
    pos-21-02 - location
    pos-21-03 - location
    pos-21-04 - location
    pos-21-05 - location
    pos-21-06 - location
    pos-21-07 - location
    pos-21-08 - location
    pos-21-09 - location
    pos-21-10 - location
    pos-22-01 - location
    pos-22-02 - location
    pos-22-03 - location
    pos-22-04 - location
    pos-22-05 - location
    pos-22-06 - location
    pos-22-07 - location
    pos-22-08 - location
    pos-22-09 - location
    pos-22-10 - location
    stone-01 - stone
    stone-02 - stone
    stone-03 - stone
    stone-04 - stone
  )
  (:init
    (IS-GOAL pos-09-06)
    (IS-GOAL pos-10-06)
    (IS-GOAL pos-11-06)
    (IS-GOAL pos-12-06)
    (IS-NONGOAL pos-01-01)
    (IS-NONGOAL pos-01-02)
    (IS-NONGOAL pos-01-03)
    (IS-NONGOAL pos-01-04)
    (IS-NONGOAL pos-01-05)
    (IS-NONGOAL pos-01-06)
    (IS-NONGOAL pos-01-07)
    (IS-NONGOAL pos-01-08)
    (IS-NONGOAL pos-01-09)
    (IS-NONGOAL pos-01-10)
    (IS-NONGOAL pos-02-01)
    (IS-NONGOAL pos-02-02)
    (IS-NONGOAL pos-02-03)
    (IS-NONGOAL pos-02-04)
    (IS-NONGOAL pos-02-05)
    (IS-NONGOAL pos-02-06)
    (IS-NONGOAL pos-02-07)
    (IS-NONGOAL pos-02-08)
    (IS-NONGOAL pos-02-09)
    (IS-NONGOAL pos-02-10)
    (IS-NONGOAL pos-03-01)
    (IS-NONGOAL pos-03-02)
    (IS-NONGOAL pos-03-03)
    (IS-NONGOAL pos-03-04)
    (IS-NONGOAL pos-03-05)
    (IS-NONGOAL pos-03-06)
    (IS-NONGOAL pos-03-07)
    (IS-NONGOAL pos-03-08)
    (IS-NONGOAL pos-03-09)
    (IS-NONGOAL pos-03-10)
    (IS-NONGOAL pos-04-01)
    (IS-NONGOAL pos-04-02)
    (IS-NONGOAL pos-04-03)
    (IS-NONGOAL pos-04-04)
    (IS-NONGOAL pos-04-05)
    (IS-NONGOAL pos-04-06)
    (IS-NONGOAL pos-04-07)
    (IS-NONGOAL pos-04-08)
    (IS-NONGOAL pos-04-09)
    (IS-NONGOAL pos-04-10)
    (IS-NONGOAL pos-05-01)
    (IS-NONGOAL pos-05-02)
    (IS-NONGOAL pos-05-03)
    (IS-NONGOAL pos-05-04)
    (IS-NONGOAL pos-05-05)
    (IS-NONGOAL pos-05-06)
    (IS-NONGOAL pos-05-07)
    (IS-NONGOAL pos-05-08)
    (IS-NONGOAL pos-05-09)
    (IS-NONGOAL pos-05-10)
    (IS-NONGOAL pos-06-01)
    (IS-NONGOAL pos-06-02)
    (IS-NONGOAL pos-06-03)
    (IS-NONGOAL pos-06-04)
    (IS-NONGOAL pos-06-05)
    (IS-NONGOAL pos-06-06)
    (IS-NONGOAL pos-06-07)
    (IS-NONGOAL pos-06-08)
    (IS-NONGOAL pos-06-09)
    (IS-NONGOAL pos-06-10)
    (IS-NONGOAL pos-07-01)
    (IS-NONGOAL pos-07-02)
    (IS-NONGOAL pos-07-03)
    (IS-NONGOAL pos-07-04)
    (IS-NONGOAL pos-07-05)
    (IS-NONGOAL pos-07-06)
    (IS-NONGOAL pos-07-07)
    (IS-NONGOAL pos-07-08)
    (IS-NONGOAL pos-07-09)
    (IS-NONGOAL pos-07-10)
    (IS-NONGOAL pos-08-01)
    (IS-NONGOAL pos-08-02)
    (IS-NONGOAL pos-08-03)
    (IS-NONGOAL pos-08-04)
    (IS-NONGOAL pos-08-05)
    (IS-NONGOAL pos-08-06)
    (IS-NONGOAL pos-08-07)
    (IS-NONGOAL pos-08-08)
    (IS-NONGOAL pos-08-09)
    (IS-NONGOAL pos-08-10)
    (IS-NONGOAL pos-09-01)
    (IS-NONGOAL pos-09-02)
    (IS-NONGOAL pos-09-03)
    (IS-NONGOAL pos-09-04)
    (IS-NONGOAL pos-09-05)
    (IS-NONGOAL pos-09-07)
    (IS-NONGOAL pos-09-08)
    (IS-NONGOAL pos-09-09)
    (IS-NONGOAL pos-09-10)
    (IS-NONGOAL pos-10-01)
    (IS-NONGOAL pos-10-02)
    (IS-NONGOAL pos-10-03)
    (IS-NONGOAL pos-10-04)
    (IS-NONGOAL pos-10-05)
    (IS-NONGOAL pos-10-07)
    (IS-NONGOAL pos-10-08)
    (IS-NONGOAL pos-10-09)
    (IS-NONGOAL pos-10-10)
    (IS-NONGOAL pos-11-01)
    (IS-NONGOAL pos-11-02)
    (IS-NONGOAL pos-11-03)
    (IS-NONGOAL pos-11-04)
    (IS-NONGOAL pos-11-05)
    (IS-NONGOAL pos-11-07)
    (IS-NONGOAL pos-11-08)
    (IS-NONGOAL pos-11-09)
    (IS-NONGOAL pos-11-10)
    (IS-NONGOAL pos-12-01)
    (IS-NONGOAL pos-12-02)
    (IS-NONGOAL pos-12-03)
    (IS-NONGOAL pos-12-04)
    (IS-NONGOAL pos-12-05)
    (IS-NONGOAL pos-12-07)
    (IS-NONGOAL pos-12-08)
    (IS-NONGOAL pos-12-09)
    (IS-NONGOAL pos-12-10)
    (IS-NONGOAL pos-13-01)
    (IS-NONGOAL pos-13-02)
    (IS-NONGOAL pos-13-03)
    (IS-NONGOAL pos-13-04)
    (IS-NONGOAL pos-13-05)
    (IS-NONGOAL pos-13-06)
    (IS-NONGOAL pos-13-07)
    (IS-NONGOAL pos-13-08)
    (IS-NONGOAL pos-13-09)
    (IS-NONGOAL pos-13-10)
    (IS-NONGOAL pos-14-01)
    (IS-NONGOAL pos-14-02)
    (IS-NONGOAL pos-14-03)
    (IS-NONGOAL pos-14-04)
    (IS-NONGOAL pos-14-05)
    (IS-NONGOAL pos-14-06)
    (IS-NONGOAL pos-14-07)
    (IS-NONGOAL pos-14-08)
    (IS-NONGOAL pos-14-09)
    (IS-NONGOAL pos-14-10)
    (IS-NONGOAL pos-15-01)
    (IS-NONGOAL pos-15-02)
    (IS-NONGOAL pos-15-03)
    (IS-NONGOAL pos-15-04)
    (IS-NONGOAL pos-15-05)
    (IS-NONGOAL pos-15-06)
    (IS-NONGOAL pos-15-07)
    (IS-NONGOAL pos-15-08)
    (IS-NONGOAL pos-15-09)
    (IS-NONGOAL pos-15-10)
    (IS-NONGOAL pos-16-01)
    (IS-NONGOAL pos-16-02)
    (IS-NONGOAL pos-16-03)
    (IS-NONGOAL pos-16-04)
    (IS-NONGOAL pos-16-05)
    (IS-NONGOAL pos-16-06)
    (IS-NONGOAL pos-16-07)
    (IS-NONGOAL pos-16-08)
    (IS-NONGOAL pos-16-09)
    (IS-NONGOAL pos-16-10)
    (IS-NONGOAL pos-17-01)
    (IS-NONGOAL pos-17-02)
    (IS-NONGOAL pos-17-03)
    (IS-NONGOAL pos-17-04)
    (IS-NONGOAL pos-17-05)
    (IS-NONGOAL pos-17-06)
    (IS-NONGOAL pos-17-07)
    (IS-NONGOAL pos-17-08)
    (IS-NONGOAL pos-17-09)
    (IS-NONGOAL pos-17-10)
    (IS-NONGOAL pos-18-01)
    (IS-NONGOAL pos-18-02)
    (IS-NONGOAL pos-18-03)
    (IS-NONGOAL pos-18-04)
    (IS-NONGOAL pos-18-05)
    (IS-NONGOAL pos-18-06)
    (IS-NONGOAL pos-18-07)
    (IS-NONGOAL pos-18-08)
    (IS-NONGOAL pos-18-09)
    (IS-NONGOAL pos-18-10)
    (IS-NONGOAL pos-19-01)
    (IS-NONGOAL pos-19-02)
    (IS-NONGOAL pos-19-03)
    (IS-NONGOAL pos-19-04)
    (IS-NONGOAL pos-19-05)
    (IS-NONGOAL pos-19-06)
    (IS-NONGOAL pos-19-07)
    (IS-NONGOAL pos-19-08)
    (IS-NONGOAL pos-19-09)
    (IS-NONGOAL pos-19-10)
    (IS-NONGOAL pos-20-01)
    (IS-NONGOAL pos-20-02)
    (IS-NONGOAL pos-20-03)
    (IS-NONGOAL pos-20-04)
    (IS-NONGOAL pos-20-05)
    (IS-NONGOAL pos-20-06)
    (IS-NONGOAL pos-20-07)
    (IS-NONGOAL pos-20-08)
    (IS-NONGOAL pos-20-09)
    (IS-NONGOAL pos-20-10)
    (IS-NONGOAL pos-21-01)
    (IS-NONGOAL pos-21-02)
    (IS-NONGOAL pos-21-03)
    (IS-NONGOAL pos-21-04)
    (IS-NONGOAL pos-21-05)
    (IS-NONGOAL pos-21-06)
    (IS-NONGOAL pos-21-07)
    (IS-NONGOAL pos-21-08)
    (IS-NONGOAL pos-21-09)
    (IS-NONGOAL pos-21-10)
    (IS-NONGOAL pos-22-01)
    (IS-NONGOAL pos-22-02)
    (IS-NONGOAL pos-22-03)
    (IS-NONGOAL pos-22-04)
    (IS-NONGOAL pos-22-05)
    (IS-NONGOAL pos-22-06)
    (IS-NONGOAL pos-22-07)
    (IS-NONGOAL pos-22-08)
    (IS-NONGOAL pos-22-09)
    (IS-NONGOAL pos-22-10)
    (MOVE-DIR pos-01-01 pos-01-02 dir-down)
    (MOVE-DIR pos-01-01 pos-02-01 dir-right)
    (MOVE-DIR pos-01-02 pos-01-01 dir-up)
    (MOVE-DIR pos-01-02 pos-02-02 dir-right)
    (MOVE-DIR pos-01-09 pos-01-10 dir-down)
    (MOVE-DIR pos-01-09 pos-02-09 dir-right)
    (MOVE-DIR pos-01-10 pos-01-09 dir-up)
    (MOVE-DIR pos-01-10 pos-02-10 dir-right)
    (MOVE-DIR pos-02-01 pos-01-01 dir-left)
    (MOVE-DIR pos-02-01 pos-02-02 dir-down)
    (MOVE-DIR pos-02-01 pos-03-01 dir-right)
    (MOVE-DIR pos-02-02 pos-01-02 dir-left)
    (MOVE-DIR pos-02-02 pos-02-01 dir-up)
    (MOVE-DIR pos-02-02 pos-03-02 dir-right)
    (MOVE-DIR pos-02-04 pos-02-05 dir-down)
    (MOVE-DIR pos-02-04 pos-03-04 dir-right)
    (MOVE-DIR pos-02-05 pos-02-04 dir-up)
    (MOVE-DIR pos-02-05 pos-02-06 dir-down)
    (MOVE-DIR pos-02-05 pos-03-05 dir-right)
    (MOVE-DIR pos-02-06 pos-02-05 dir-up)
    (MOVE-DIR pos-02-06 pos-02-07 dir-down)
    (MOVE-DIR pos-02-06 pos-03-06 dir-right)
    (MOVE-DIR pos-02-07 pos-02-06 dir-up)
    (MOVE-DIR pos-02-07 pos-03-07 dir-right)
    (MOVE-DIR pos-02-09 pos-01-09 dir-left)
    (MOVE-DIR pos-02-09 pos-02-10 dir-down)
    (MOVE-DIR pos-02-09 pos-03-09 dir-right)
    (MOVE-DIR pos-02-10 pos-01-10 dir-left)
    (MOVE-DIR pos-02-10 pos-02-09 dir-up)
    (MOVE-DIR pos-02-10 pos-03-10 dir-right)
    (MOVE-DIR pos-03-01 pos-02-01 dir-left)
    (MOVE-DIR pos-03-01 pos-03-02 dir-down)
    (MOVE-DIR pos-03-01 pos-04-01 dir-right)
    (MOVE-DIR pos-03-02 pos-02-02 dir-left)
    (MOVE-DIR pos-03-02 pos-03-01 dir-up)
    (MOVE-DIR pos-03-02 pos-04-02 dir-right)
    (MOVE-DIR pos-03-04 pos-02-04 dir-left)
    (MOVE-DIR pos-03-04 pos-03-05 dir-down)
    (MOVE-DIR pos-03-04 pos-04-04 dir-right)
    (MOVE-DIR pos-03-05 pos-02-05 dir-left)
    (MOVE-DIR pos-03-05 pos-03-04 dir-up)
    (MOVE-DIR pos-03-05 pos-03-06 dir-down)
    (MOVE-DIR pos-03-05 pos-04-05 dir-right)
    (MOVE-DIR pos-03-06 pos-02-06 dir-left)
    (MOVE-DIR pos-03-06 pos-03-05 dir-up)
    (MOVE-DIR pos-03-06 pos-03-07 dir-down)
    (MOVE-DIR pos-03-07 pos-02-07 dir-left)
    (MOVE-DIR pos-03-07 pos-03-06 dir-up)
    (MOVE-DIR pos-03-07 pos-04-07 dir-right)
    (MOVE-DIR pos-03-09 pos-02-09 dir-left)
    (MOVE-DIR pos-03-09 pos-03-10 dir-down)
    (MOVE-DIR pos-03-09 pos-04-09 dir-right)
    (MOVE-DIR pos-03-10 pos-02-10 dir-left)
    (MOVE-DIR pos-03-10 pos-03-09 dir-up)
    (MOVE-DIR pos-03-10 pos-04-10 dir-right)
    (MOVE-DIR pos-04-01 pos-03-01 dir-left)
    (MOVE-DIR pos-04-01 pos-04-02 dir-down)
    (MOVE-DIR pos-04-01 pos-05-01 dir-right)
    (MOVE-DIR pos-04-02 pos-03-02 dir-left)
    (MOVE-DIR pos-04-02 pos-04-01 dir-up)
    (MOVE-DIR pos-04-02 pos-05-02 dir-right)
    (MOVE-DIR pos-04-04 pos-03-04 dir-left)
    (MOVE-DIR pos-04-04 pos-04-05 dir-down)
    (MOVE-DIR pos-04-04 pos-05-04 dir-right)
    (MOVE-DIR pos-04-05 pos-03-05 dir-left)
    (MOVE-DIR pos-04-05 pos-04-04 dir-up)
    (MOVE-DIR pos-04-05 pos-05-05 dir-right)
    (MOVE-DIR pos-04-07 pos-03-07 dir-left)
    (MOVE-DIR pos-04-07 pos-05-07 dir-right)
    (MOVE-DIR pos-04-09 pos-03-09 dir-left)
    (MOVE-DIR pos-04-09 pos-04-10 dir-down)
    (MOVE-DIR pos-04-09 pos-05-09 dir-right)
    (MOVE-DIR pos-04-10 pos-03-10 dir-left)
    (MOVE-DIR pos-04-10 pos-04-09 dir-up)
    (MOVE-DIR pos-04-10 pos-05-10 dir-right)
    (MOVE-DIR pos-05-01 pos-04-01 dir-left)
    (MOVE-DIR pos-05-01 pos-05-02 dir-down)
    (MOVE-DIR pos-05-01 pos-06-01 dir-right)
    (MOVE-DIR pos-05-02 pos-04-02 dir-left)
    (MOVE-DIR pos-05-02 pos-05-01 dir-up)
    (MOVE-DIR pos-05-02 pos-06-02 dir-right)
    (MOVE-DIR pos-05-04 pos-04-04 dir-left)
    (MOVE-DIR pos-05-04 pos-05-05 dir-down)
    (MOVE-DIR pos-05-04 pos-06-04 dir-right)
    (MOVE-DIR pos-05-05 pos-04-05 dir-left)
    (MOVE-DIR pos-05-05 pos-05-04 dir-up)
    (MOVE-DIR pos-05-05 pos-05-06 dir-down)
    (MOVE-DIR pos-05-05 pos-06-05 dir-right)
    (MOVE-DIR pos-05-06 pos-05-05 dir-up)
    (MOVE-DIR pos-05-06 pos-05-07 dir-down)
    (MOVE-DIR pos-05-06 pos-06-06 dir-right)
    (MOVE-DIR pos-05-07 pos-04-07 dir-left)
    (MOVE-DIR pos-05-07 pos-05-06 dir-up)
    (MOVE-DIR pos-05-09 pos-04-09 dir-left)
    (MOVE-DIR pos-05-09 pos-05-10 dir-down)
    (MOVE-DIR pos-05-09 pos-06-09 dir-right)
    (MOVE-DIR pos-05-10 pos-04-10 dir-left)
    (MOVE-DIR pos-05-10 pos-05-09 dir-up)
    (MOVE-DIR pos-05-10 pos-06-10 dir-right)
    (MOVE-DIR pos-06-01 pos-05-01 dir-left)
    (MOVE-DIR pos-06-01 pos-06-02 dir-down)
    (MOVE-DIR pos-06-01 pos-07-01 dir-right)
    (MOVE-DIR pos-06-02 pos-05-02 dir-left)
    (MOVE-DIR pos-06-02 pos-06-01 dir-up)
    (MOVE-DIR pos-06-02 pos-07-02 dir-right)
    (MOVE-DIR pos-06-04 pos-05-04 dir-left)
    (MOVE-DIR pos-06-04 pos-06-05 dir-down)
    (MOVE-DIR pos-06-05 pos-05-05 dir-left)
    (MOVE-DIR pos-06-05 pos-06-04 dir-up)
    (MOVE-DIR pos-06-05 pos-06-06 dir-down)
    (MOVE-DIR pos-06-06 pos-05-06 dir-left)
    (MOVE-DIR pos-06-06 pos-06-05 dir-up)
    (MOVE-DIR pos-06-06 pos-07-06 dir-right)
    (MOVE-DIR pos-06-09 pos-05-09 dir-left)
    (MOVE-DIR pos-06-09 pos-06-10 dir-down)
    (MOVE-DIR pos-06-09 pos-07-09 dir-right)
    (MOVE-DIR pos-06-10 pos-05-10 dir-left)
    (MOVE-DIR pos-06-10 pos-06-09 dir-up)
    (MOVE-DIR pos-06-10 pos-07-10 dir-right)
    (MOVE-DIR pos-07-01 pos-06-01 dir-left)
    (MOVE-DIR pos-07-01 pos-07-02 dir-down)
    (MOVE-DIR pos-07-01 pos-08-01 dir-right)
    (MOVE-DIR pos-07-02 pos-06-02 dir-left)
    (MOVE-DIR pos-07-02 pos-07-01 dir-up)
    (MOVE-DIR pos-07-02 pos-08-02 dir-right)
    (MOVE-DIR pos-07-06 pos-06-06 dir-left)
    (MOVE-DIR pos-07-06 pos-08-06 dir-right)
    (MOVE-DIR pos-07-08 pos-07-09 dir-down)
    (MOVE-DIR pos-07-08 pos-08-08 dir-right)
    (MOVE-DIR pos-07-09 pos-06-09 dir-left)
    (MOVE-DIR pos-07-09 pos-07-08 dir-up)
    (MOVE-DIR pos-07-09 pos-07-10 dir-down)
    (MOVE-DIR pos-07-09 pos-08-09 dir-right)
    (MOVE-DIR pos-07-10 pos-06-10 dir-left)
    (MOVE-DIR pos-07-10 pos-07-09 dir-up)
    (MOVE-DIR pos-07-10 pos-08-10 dir-right)
    (MOVE-DIR pos-08-01 pos-07-01 dir-left)
    (MOVE-DIR pos-08-01 pos-08-02 dir-down)
    (MOVE-DIR pos-08-01 pos-09-01 dir-right)
    (MOVE-DIR pos-08-02 pos-07-02 dir-left)
    (MOVE-DIR pos-08-02 pos-08-01 dir-up)
    (MOVE-DIR pos-08-02 pos-08-03 dir-down)
    (MOVE-DIR pos-08-02 pos-09-02 dir-right)
    (MOVE-DIR pos-08-03 pos-08-02 dir-up)
    (MOVE-DIR pos-08-03 pos-08-04 dir-down)
    (MOVE-DIR pos-08-03 pos-09-03 dir-right)
    (MOVE-DIR pos-08-04 pos-08-03 dir-up)
    (MOVE-DIR pos-08-04 pos-09-04 dir-right)
    (MOVE-DIR pos-08-06 pos-07-06 dir-left)
    (MOVE-DIR pos-08-06 pos-09-06 dir-right)
    (MOVE-DIR pos-08-08 pos-07-08 dir-left)
    (MOVE-DIR pos-08-08 pos-08-09 dir-down)
    (MOVE-DIR pos-08-08 pos-09-08 dir-right)
    (MOVE-DIR pos-08-09 pos-07-09 dir-left)
    (MOVE-DIR pos-08-09 pos-08-08 dir-up)
    (MOVE-DIR pos-08-09 pos-08-10 dir-down)
    (MOVE-DIR pos-08-09 pos-09-09 dir-right)
    (MOVE-DIR pos-08-10 pos-07-10 dir-left)
    (MOVE-DIR pos-08-10 pos-08-09 dir-up)
    (MOVE-DIR pos-08-10 pos-09-10 dir-right)
    (MOVE-DIR pos-09-01 pos-08-01 dir-left)
    (MOVE-DIR pos-09-01 pos-09-02 dir-down)
    (MOVE-DIR pos-09-01 pos-10-01 dir-right)
    (MOVE-DIR pos-09-02 pos-08-02 dir-left)
    (MOVE-DIR pos-09-02 pos-09-01 dir-up)
    (MOVE-DIR pos-09-02 pos-09-03 dir-down)
    (MOVE-DIR pos-09-02 pos-10-02 dir-right)
    (MOVE-DIR pos-09-03 pos-08-03 dir-left)
    (MOVE-DIR pos-09-03 pos-09-02 dir-up)
    (MOVE-DIR pos-09-03 pos-09-04 dir-down)
    (MOVE-DIR pos-09-04 pos-08-04 dir-left)
    (MOVE-DIR pos-09-04 pos-09-03 dir-up)
    (MOVE-DIR pos-09-06 pos-08-06 dir-left)
    (MOVE-DIR pos-09-06 pos-10-06 dir-right)
    (MOVE-DIR pos-09-08 pos-08-08 dir-left)
    (MOVE-DIR pos-09-08 pos-09-09 dir-down)
    (MOVE-DIR pos-09-09 pos-08-09 dir-left)
    (MOVE-DIR pos-09-09 pos-09-08 dir-up)
    (MOVE-DIR pos-09-09 pos-09-10 dir-down)
    (MOVE-DIR pos-09-10 pos-08-10 dir-left)
    (MOVE-DIR pos-09-10 pos-09-09 dir-up)
    (MOVE-DIR pos-10-01 pos-09-01 dir-left)
    (MOVE-DIR pos-10-01 pos-10-02 dir-down)
    (MOVE-DIR pos-10-01 pos-11-01 dir-right)
    (MOVE-DIR pos-10-02 pos-09-02 dir-left)
    (MOVE-DIR pos-10-02 pos-10-01 dir-up)
    (MOVE-DIR pos-10-02 pos-11-02 dir-right)
    (MOVE-DIR pos-10-06 pos-09-06 dir-left)
    (MOVE-DIR pos-10-06 pos-11-06 dir-right)
    (MOVE-DIR pos-11-01 pos-10-01 dir-left)
    (MOVE-DIR pos-11-01 pos-11-02 dir-down)
    (MOVE-DIR pos-11-01 pos-12-01 dir-right)
    (MOVE-DIR pos-11-02 pos-10-02 dir-left)
    (MOVE-DIR pos-11-02 pos-11-01 dir-up)
    (MOVE-DIR pos-11-02 pos-12-02 dir-right)
    (MOVE-DIR pos-11-04 pos-11-05 dir-down)
    (MOVE-DIR pos-11-04 pos-12-04 dir-right)
    (MOVE-DIR pos-11-05 pos-11-04 dir-up)
    (MOVE-DIR pos-11-05 pos-11-06 dir-down)
    (MOVE-DIR pos-11-05 pos-12-05 dir-right)
    (MOVE-DIR pos-11-06 pos-10-06 dir-left)
    (MOVE-DIR pos-11-06 pos-11-05 dir-up)
    (MOVE-DIR pos-11-06 pos-11-07 dir-down)
    (MOVE-DIR pos-11-06 pos-12-06 dir-right)
    (MOVE-DIR pos-11-07 pos-11-06 dir-up)
    (MOVE-DIR pos-11-07 pos-11-08 dir-down)
    (MOVE-DIR pos-11-08 pos-11-07 dir-up)
    (MOVE-DIR pos-11-08 pos-11-09 dir-down)
    (MOVE-DIR pos-11-08 pos-12-08 dir-right)
    (MOVE-DIR pos-11-09 pos-11-08 dir-up)
    (MOVE-DIR pos-11-09 pos-12-09 dir-right)
    (MOVE-DIR pos-12-01 pos-11-01 dir-left)
    (MOVE-DIR pos-12-01 pos-12-02 dir-down)
    (MOVE-DIR pos-12-01 pos-13-01 dir-right)
    (MOVE-DIR pos-12-02 pos-11-02 dir-left)
    (MOVE-DIR pos-12-02 pos-12-01 dir-up)
    (MOVE-DIR pos-12-02 pos-13-02 dir-right)
    (MOVE-DIR pos-12-04 pos-11-04 dir-left)
    (MOVE-DIR pos-12-04 pos-12-05 dir-down)
    (MOVE-DIR pos-12-05 pos-11-05 dir-left)
    (MOVE-DIR pos-12-05 pos-12-04 dir-up)
    (MOVE-DIR pos-12-05 pos-12-06 dir-down)
    (MOVE-DIR pos-12-06 pos-11-06 dir-left)
    (MOVE-DIR pos-12-06 pos-12-05 dir-up)
    (MOVE-DIR pos-12-08 pos-11-08 dir-left)
    (MOVE-DIR pos-12-08 pos-12-09 dir-down)
    (MOVE-DIR pos-12-08 pos-13-08 dir-right)
    (MOVE-DIR pos-12-09 pos-11-09 dir-left)
    (MOVE-DIR pos-12-09 pos-12-08 dir-up)
    (MOVE-DIR pos-13-01 pos-12-01 dir-left)
    (MOVE-DIR pos-13-01 pos-13-02 dir-down)
    (MOVE-DIR pos-13-01 pos-14-01 dir-right)
    (MOVE-DIR pos-13-02 pos-12-02 dir-left)
    (MOVE-DIR pos-13-02 pos-13-01 dir-up)
    (MOVE-DIR pos-13-02 pos-14-02 dir-right)
    (MOVE-DIR pos-13-08 pos-12-08 dir-left)
    (MOVE-DIR pos-13-08 pos-14-08 dir-right)
    (MOVE-DIR pos-14-01 pos-13-01 dir-left)
    (MOVE-DIR pos-14-01 pos-14-02 dir-down)
    (MOVE-DIR pos-14-01 pos-15-01 dir-right)
    (MOVE-DIR pos-14-02 pos-13-02 dir-left)
    (MOVE-DIR pos-14-02 pos-14-01 dir-up)
    (MOVE-DIR pos-14-02 pos-15-02 dir-right)
    (MOVE-DIR pos-14-04 pos-14-05 dir-down)
    (MOVE-DIR pos-14-04 pos-15-04 dir-right)
    (MOVE-DIR pos-14-05 pos-14-04 dir-up)
    (MOVE-DIR pos-14-05 pos-15-05 dir-right)
    (MOVE-DIR pos-14-07 pos-14-08 dir-down)
    (MOVE-DIR pos-14-07 pos-15-07 dir-right)
    (MOVE-DIR pos-14-08 pos-13-08 dir-left)
    (MOVE-DIR pos-14-08 pos-14-07 dir-up)
    (MOVE-DIR pos-14-08 pos-15-08 dir-right)
    (MOVE-DIR pos-14-10 pos-15-10 dir-right)
    (MOVE-DIR pos-15-01 pos-14-01 dir-left)
    (MOVE-DIR pos-15-01 pos-15-02 dir-down)
    (MOVE-DIR pos-15-02 pos-14-02 dir-left)
    (MOVE-DIR pos-15-02 pos-15-01 dir-up)
    (MOVE-DIR pos-15-04 pos-14-04 dir-left)
    (MOVE-DIR pos-15-04 pos-15-05 dir-down)
    (MOVE-DIR pos-15-04 pos-16-04 dir-right)
    (MOVE-DIR pos-15-05 pos-14-05 dir-left)
    (MOVE-DIR pos-15-05 pos-15-04 dir-up)
    (MOVE-DIR pos-15-05 pos-15-06 dir-down)
    (MOVE-DIR pos-15-05 pos-16-05 dir-right)
    (MOVE-DIR pos-15-06 pos-15-05 dir-up)
    (MOVE-DIR pos-15-06 pos-15-07 dir-down)
    (MOVE-DIR pos-15-07 pos-14-07 dir-left)
    (MOVE-DIR pos-15-07 pos-15-06 dir-up)
    (MOVE-DIR pos-15-07 pos-15-08 dir-down)
    (MOVE-DIR pos-15-07 pos-16-07 dir-right)
    (MOVE-DIR pos-15-08 pos-14-08 dir-left)
    (MOVE-DIR pos-15-08 pos-15-07 dir-up)
    (MOVE-DIR pos-15-08 pos-16-08 dir-right)
    (MOVE-DIR pos-15-10 pos-14-10 dir-left)
    (MOVE-DIR pos-15-10 pos-16-10 dir-right)
    (MOVE-DIR pos-16-04 pos-15-04 dir-left)
    (MOVE-DIR pos-16-04 pos-16-05 dir-down)
    (MOVE-DIR pos-16-04 pos-17-04 dir-right)
    (MOVE-DIR pos-16-05 pos-15-05 dir-left)
    (MOVE-DIR pos-16-05 pos-16-04 dir-up)
    (MOVE-DIR pos-16-05 pos-17-05 dir-right)
    (MOVE-DIR pos-16-07 pos-15-07 dir-left)
    (MOVE-DIR pos-16-07 pos-16-08 dir-down)
    (MOVE-DIR pos-16-07 pos-17-07 dir-right)
    (MOVE-DIR pos-16-08 pos-15-08 dir-left)
    (MOVE-DIR pos-16-08 pos-16-07 dir-up)
    (MOVE-DIR pos-16-08 pos-17-08 dir-right)
    (MOVE-DIR pos-16-10 pos-15-10 dir-left)
    (MOVE-DIR pos-16-10 pos-17-10 dir-right)
    (MOVE-DIR pos-17-02 pos-17-03 dir-down)
    (MOVE-DIR pos-17-02 pos-18-02 dir-right)
    (MOVE-DIR pos-17-03 pos-17-02 dir-up)
    (MOVE-DIR pos-17-03 pos-17-04 dir-down)
    (MOVE-DIR pos-17-04 pos-16-04 dir-left)
    (MOVE-DIR pos-17-04 pos-17-03 dir-up)
    (MOVE-DIR pos-17-04 pos-17-05 dir-down)
    (MOVE-DIR pos-17-04 pos-18-04 dir-right)
    (MOVE-DIR pos-17-05 pos-16-05 dir-left)
    (MOVE-DIR pos-17-05 pos-17-04 dir-up)
    (MOVE-DIR pos-17-05 pos-18-05 dir-right)
    (MOVE-DIR pos-17-07 pos-16-07 dir-left)
    (MOVE-DIR pos-17-07 pos-17-08 dir-down)
    (MOVE-DIR pos-17-07 pos-18-07 dir-right)
    (MOVE-DIR pos-17-08 pos-16-08 dir-left)
    (MOVE-DIR pos-17-08 pos-17-07 dir-up)
    (MOVE-DIR pos-17-08 pos-18-08 dir-right)
    (MOVE-DIR pos-17-10 pos-16-10 dir-left)
    (MOVE-DIR pos-17-10 pos-18-10 dir-right)
    (MOVE-DIR pos-18-02 pos-17-02 dir-left)
    (MOVE-DIR pos-18-02 pos-19-02 dir-right)
    (MOVE-DIR pos-18-04 pos-17-04 dir-left)
    (MOVE-DIR pos-18-04 pos-18-05 dir-down)
    (MOVE-DIR pos-18-04 pos-19-04 dir-right)
    (MOVE-DIR pos-18-05 pos-17-05 dir-left)
    (MOVE-DIR pos-18-05 pos-18-04 dir-up)
    (MOVE-DIR pos-18-07 pos-17-07 dir-left)
    (MOVE-DIR pos-18-07 pos-18-08 dir-down)
    (MOVE-DIR pos-18-07 pos-19-07 dir-right)
    (MOVE-DIR pos-18-08 pos-17-08 dir-left)
    (MOVE-DIR pos-18-08 pos-18-07 dir-up)
    (MOVE-DIR pos-18-08 pos-19-08 dir-right)
    (MOVE-DIR pos-18-10 pos-17-10 dir-left)
    (MOVE-DIR pos-18-10 pos-19-10 dir-right)
    (MOVE-DIR pos-19-02 pos-18-02 dir-left)
    (MOVE-DIR pos-19-02 pos-19-03 dir-down)
    (MOVE-DIR pos-19-03 pos-19-02 dir-up)
    (MOVE-DIR pos-19-03 pos-19-04 dir-down)
    (MOVE-DIR pos-19-04 pos-18-04 dir-left)
    (MOVE-DIR pos-19-04 pos-19-03 dir-up)
    (MOVE-DIR pos-19-07 pos-18-07 dir-left)
    (MOVE-DIR pos-19-07 pos-19-08 dir-down)
    (MOVE-DIR pos-19-07 pos-20-07 dir-right)
    (MOVE-DIR pos-19-08 pos-18-08 dir-left)
    (MOVE-DIR pos-19-08 pos-19-07 dir-up)
    (MOVE-DIR pos-19-08 pos-20-08 dir-right)
    (MOVE-DIR pos-19-10 pos-18-10 dir-left)
    (MOVE-DIR pos-19-10 pos-20-10 dir-right)
    (MOVE-DIR pos-20-06 pos-20-07 dir-down)
    (MOVE-DIR pos-20-06 pos-21-06 dir-right)
    (MOVE-DIR pos-20-07 pos-19-07 dir-left)
    (MOVE-DIR pos-20-07 pos-20-06 dir-up)
    (MOVE-DIR pos-20-07 pos-20-08 dir-down)
    (MOVE-DIR pos-20-07 pos-21-07 dir-right)
    (MOVE-DIR pos-20-08 pos-19-08 dir-left)
    (MOVE-DIR pos-20-08 pos-20-07 dir-up)
    (MOVE-DIR pos-20-08 pos-21-08 dir-right)
    (MOVE-DIR pos-20-10 pos-19-10 dir-left)
    (MOVE-DIR pos-20-10 pos-21-10 dir-right)
    (MOVE-DIR pos-21-01 pos-21-02 dir-down)
    (MOVE-DIR pos-21-01 pos-22-01 dir-right)
    (MOVE-DIR pos-21-02 pos-21-01 dir-up)
    (MOVE-DIR pos-21-02 pos-21-03 dir-down)
    (MOVE-DIR pos-21-02 pos-22-02 dir-right)
    (MOVE-DIR pos-21-03 pos-21-02 dir-up)
    (MOVE-DIR pos-21-03 pos-21-04 dir-down)
    (MOVE-DIR pos-21-03 pos-22-03 dir-right)
    (MOVE-DIR pos-21-04 pos-21-03 dir-up)
    (MOVE-DIR pos-21-04 pos-22-04 dir-right)
    (MOVE-DIR pos-21-06 pos-20-06 dir-left)
    (MOVE-DIR pos-21-06 pos-21-07 dir-down)
    (MOVE-DIR pos-21-07 pos-20-07 dir-left)
    (MOVE-DIR pos-21-07 pos-21-06 dir-up)
    (MOVE-DIR pos-21-07 pos-21-08 dir-down)
    (MOVE-DIR pos-21-08 pos-20-08 dir-left)
    (MOVE-DIR pos-21-08 pos-21-07 dir-up)
    (MOVE-DIR pos-21-10 pos-20-10 dir-left)
    (MOVE-DIR pos-21-10 pos-22-10 dir-right)
    (MOVE-DIR pos-22-01 pos-21-01 dir-left)
    (MOVE-DIR pos-22-01 pos-22-02 dir-down)
    (MOVE-DIR pos-22-02 pos-21-02 dir-left)
    (MOVE-DIR pos-22-02 pos-22-01 dir-up)
    (MOVE-DIR pos-22-02 pos-22-03 dir-down)
    (MOVE-DIR pos-22-03 pos-21-03 dir-left)
    (MOVE-DIR pos-22-03 pos-22-02 dir-up)
    (MOVE-DIR pos-22-03 pos-22-04 dir-down)
    (MOVE-DIR pos-22-04 pos-21-04 dir-left)
    (MOVE-DIR pos-22-04 pos-22-03 dir-up)
    (MOVE-DIR pos-22-10 pos-21-10 dir-left)
    (at player-01 pos-04-05)
    (at stone-01 pos-15-07)
    (at stone-02 pos-16-07)
    (at stone-03 pos-18-07)
    (at stone-04 pos-20-07)
    (clear pos-01-01)
    (clear pos-01-02)
    (clear pos-01-09)
    (clear pos-01-10)
    (clear pos-02-01)
    (clear pos-02-02)
    (clear pos-02-04)
    (clear pos-02-05)
    (clear pos-02-06)
    (clear pos-02-07)
    (clear pos-02-09)
    (clear pos-02-10)
    (clear pos-03-01)
    (clear pos-03-02)
    (clear pos-03-04)
    (clear pos-03-05)
    (clear pos-03-06)
    (clear pos-03-07)
    (clear pos-03-09)
    (clear pos-03-10)
    (clear pos-04-01)
    (clear pos-04-02)
    (clear pos-04-04)
    (clear pos-04-07)
    (clear pos-04-09)
    (clear pos-04-10)
    (clear pos-05-01)
    (clear pos-05-02)
    (clear pos-05-04)
    (clear pos-05-05)
    (clear pos-05-06)
    (clear pos-05-07)
    (clear pos-05-09)
    (clear pos-05-10)
    (clear pos-06-01)
    (clear pos-06-02)
    (clear pos-06-04)
    (clear pos-06-05)
    (clear pos-06-06)
    (clear pos-06-09)
    (clear pos-06-10)
    (clear pos-07-01)
    (clear pos-07-02)
    (clear pos-07-06)
    (clear pos-07-08)
    (clear pos-07-09)
    (clear pos-07-10)
    (clear pos-08-01)
    (clear pos-08-02)
    (clear pos-08-03)
    (clear pos-08-04)
    (clear pos-08-06)
    (clear pos-08-08)
    (clear pos-08-09)
    (clear pos-08-10)
    (clear pos-09-01)
    (clear pos-09-02)
    (clear pos-09-03)
    (clear pos-09-04)
    (clear pos-09-06)
    (clear pos-09-08)
    (clear pos-09-09)
    (clear pos-09-10)
    (clear pos-10-01)
    (clear pos-10-02)
    (clear pos-10-06)
    (clear pos-11-01)
    (clear pos-11-02)
    (clear pos-11-04)
    (clear pos-11-05)
    (clear pos-11-06)
    (clear pos-11-07)
    (clear pos-11-08)
    (clear pos-11-09)
    (clear pos-12-01)
    (clear pos-12-02)
    (clear pos-12-04)
    (clear pos-12-05)
    (clear pos-12-06)
    (clear pos-12-08)
    (clear pos-12-09)
    (clear pos-13-01)
    (clear pos-13-02)
    (clear pos-13-08)
    (clear pos-14-01)
    (clear pos-14-02)
    (clear pos-14-04)
    (clear pos-14-05)
    (clear pos-14-07)
    (clear pos-14-08)
    (clear pos-14-10)
    (clear pos-15-01)
    (clear pos-15-02)
    (clear pos-15-04)
    (clear pos-15-05)
    (clear pos-15-06)
    (clear pos-15-08)
    (clear pos-15-10)
    (clear pos-16-04)
    (clear pos-16-05)
    (clear pos-16-08)
    (clear pos-16-10)
    (clear pos-17-02)
    (clear pos-17-03)
    (clear pos-17-04)
    (clear pos-17-05)
    (clear pos-17-07)
    (clear pos-17-08)
    (clear pos-17-10)
    (clear pos-18-02)
    (clear pos-18-04)
    (clear pos-18-05)
    (clear pos-18-08)
    (clear pos-18-10)
    (clear pos-19-02)
    (clear pos-19-03)
    (clear pos-19-04)
    (clear pos-19-07)
    (clear pos-19-08)
    (clear pos-19-10)
    (clear pos-20-06)
    (clear pos-20-08)
    (clear pos-20-10)
    (clear pos-21-01)
    (clear pos-21-02)
    (clear pos-21-03)
    (clear pos-21-04)
    (clear pos-21-06)
    (clear pos-21-07)
    (clear pos-21-08)
    (clear pos-21-10)
    (clear pos-22-01)
    (clear pos-22-02)
    (clear pos-22-03)
    (clear pos-22-04)
    (clear pos-22-10)
    (= (total-cost) 0)
  )
  (:goal (and
    (at-goal stone-01)
    (at-goal stone-02)
    (at-goal stone-03)
    (at-goal stone-04)
  ))
  (:metric minimize (total-cost))
)