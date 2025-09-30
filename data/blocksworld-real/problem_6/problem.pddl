(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        yellow - block
        purple - block
        blue - block
        red - block
        green - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (clear orange)
        (ontable yellow)
        (clear yellow)
        (ontable purple)
        (on blue purple)
        (clear blue)
        (ontable red)
        (clear red)
        (ontable green)
        (clear green)
        ; orange
        ; yellow
        ; purple -> blue
        ; red
        ; green
    )
    (:goal (and (on green yellow)
        (on purple green)
        (on red purple)
        (on orange red)
        (on blue orange)))
)
    