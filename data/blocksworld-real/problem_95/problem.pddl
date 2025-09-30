(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        red - block
        yellow - block
        green - block
        orange - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on red purple)
        (clear red)
        (ontable yellow)
        (clear yellow)
        (ontable green)
        (on orange green)
        (on blue orange)
        (clear blue)
        ; purple -> red
        ; yellow
        ; green -> orange -> blue
    )
    (:goal (and (on green red)
        (on purple yellow)
        (on orange purple)
        (on blue orange)))
)
    