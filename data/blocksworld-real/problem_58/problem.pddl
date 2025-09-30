(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        yellow - block
        blue - block
        red - block
        purple - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable yellow)
        (clear yellow)
        (ontable blue)
        (clear blue)
        (ontable red)
        (clear red)
        (ontable purple)
        (on orange purple)
        (clear orange)
        ; green
        ; yellow
        ; blue
        ; red
        ; purple -> orange
    )
    (:goal (and (on green yellow)
        (on blue orange)
        (on purple red)))
)
    